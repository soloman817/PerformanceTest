using System;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Mono.CSharp.Linq;

namespace PerformanceCUDAfy
{
    class Program
    {
        private const int TileSize = 32;

        [Cudafy]
        public static void SimpleMultiplyKernel(GThread thread, float[,] a, float[,] b, float[,] c)
        {
            var aRows = a.GetLength(0);
            var bCols = b.GetLength(1);
            var aCols_bRows = a.GetLength(1);
            var row = thread.blockDim.y * thread.blockIdx.y + thread.threadIdx.y;
            var col = thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x;
            if (row >= aRows || col >= bCols) return;
            var sum = 0f;
            for (var k = 0; k < aCols_bRows; k++)
            {
                sum += a[row, k] * b[k, col];
            }
            c[row, col] = sum;
        }

        public static void SimpleMultiply()
        {
            for (var iter = 1; iter <= 3; ++iter)
            {
                Console.WriteLine("====> Test SimpleMultiply with CUDAfy C# (#.{0}) <====", iter);

                var timer = Stopwatch.StartNew();
                var gpu = CudafyHost.GetDevice();
                Console.WriteLine("GPU: {0}", gpu.GetDeviceProperties().Name);
                timer.Stop();
                Console.WriteLine("Step 1) Runtime setup                  {0} ms", timer.Elapsed.TotalMilliseconds);

                timer.Restart();
                var module = CudafyTranslator.Cudafy();
                timer.Stop();
                Console.WriteLine("Step 2) Compile                        {0} ms", timer.Elapsed.TotalMilliseconds);

                timer.Restart();
                gpu.LoadModule(module);
                timer.Stop();
                Console.WriteLine("Step 3) Load module                    {0} ms", timer.Elapsed.TotalMilliseconds);

                const int factor = 8;
                var a = Util.RandomMatrix(100 * factor, 200 * factor);
                var b = Util.RandomMatrix(200 * factor, 300 * factor);
                var c = new float[a.GetLength(0), b.GetLength(1)];

                var devA = gpu.CopyToDevice(a);
                var devB = gpu.CopyToDevice(b);
                var devC = gpu.Allocate<float>(a.GetLength(0), b.GetLength(1));

                var gridDim = new dim3(Util.Divup(b.GetLength(1), TileSize), Util.Divup(a.GetLength(0), TileSize));
                var blockDim = new dim3(TileSize, TileSize);

                // measure first kernel execution, need sync worker
                timer.Restart();
                gpu.Launch(gridDim, blockDim, SimpleMultiplyKernel, devA, devB, devC);
                gpu.Synchronize();
                timer.Stop();
                Console.WriteLine("Kernel launch first time               {0} ms", timer.Elapsed.TotalMilliseconds);

                // launch 50 kernels, and sync at last (1 sync only)
                const int repetitions = 50;
                timer.Restart();
                for (var i = 0; i < repetitions; ++i)
                {
                    gpu.Launch(gridDim, blockDim, SimpleMultiplyKernel, devA, devB, devC);
                }
                gpu.Synchronize();
                timer.Stop();
                Console.WriteLine("Kernel launch average time             {0} ms", timer.Elapsed.TotalMilliseconds / (float)repetitions);

                gpu.CopyFromDevice(devC, c);
                gpu.Free(devA);
                gpu.Free(devB);
                gpu.Free(devC);
                Util.VerifyResult(a, b, c);

            }
        }

        public static void Main()
        {
            SimpleMultiply();
        }
    }
}
