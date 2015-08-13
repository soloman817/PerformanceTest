using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CUDA;
using Alea.CUDA.IL;

namespace PerformanceAleaCS
{
    class MethodUsageJIT
    {
        private const int TileSize = 32;

        static void SimpleMultiplyKernel(deviceptr<float> a, deviceptr<float> b, deviceptr<float> c,
            int aRows, int bCols, int aCols_bRows)
        {
            var row = blockDim.y * blockIdx.y + threadIdx.y;
            var col = blockDim.x * blockIdx.x + threadIdx.x;
            if (row >= aRows || col >= bCols) return;
            var sum = 0.0f;
            for (var k = 0; k < aCols_bRows; ++k)
            {
                sum += a[row * aCols_bRows + k] * b[k * bCols + col];
            }
            c[row * bCols + col] = sum;
        }

        static public void TestSimpleMultiply()
        {
            for (var iter = 1; iter <= 3; ++iter)
            {
                Console.WriteLine("====> Test SimpleMultiply with Alea GPU C# JIT method usage (#.{0}) <====", iter);

                var timer = Stopwatch.StartNew();
                var worker = Util.Worker;
                Console.WriteLine("GPU: {0}", worker.Device.Name);
                timer.Stop();
                Console.WriteLine("Step 1) Runtime setup                   {0} ms", timer.Elapsed.TotalMilliseconds);

                const int factor = 8;
                var a = Util.RandomMatrix(100 * factor, 200 * factor);
                var b = Util.RandomMatrix(200 * factor, 300 * factor);
                var aRows = 100 * factor;
                var bCols = 300 * factor;
                var aCols_bRows = 200 * factor;
                var gridDim = new dim3(Util.Divup(bCols, TileSize), Util.Divup(aRows, TileSize));
                var blockDim = new dim3(TileSize, TileSize);
                var lp = new LaunchParam(gridDim, blockDim);

                using (var devA = worker.Malloc(a))
                using (var devB = worker.Malloc(b))
                using (var devC = worker.Malloc<float>(aRows * bCols))
                {
                    timer.Restart();
                    worker.Launch(SimpleMultiplyKernel, lp, devA.Ptr, devB.Ptr, devC.Ptr, aRows, bCols, aCols_bRows);
                    worker.Synchronize();
                    timer.Stop();
                    Console.WriteLine("Kernel launch first time                {0} ms", timer.Elapsed.TotalMilliseconds);

                    const int repetitions = 50;
                    timer.Restart();
                    for (var i = 0; i < repetitions; ++i)
                    {
                        worker.Launch(SimpleMultiplyKernel, lp, devA.Ptr, devB.Ptr, devC.Ptr, aRows, bCols, aCols_bRows);
                    }
                    worker.Synchronize();
                    timer.Stop();
                    Console.WriteLine("Kernel launch average time              {0} ms", (timer.Elapsed.TotalMilliseconds / (float)repetitions));

                    var c = devC.Gather();
                    Util.VerifyResult(a, b, c, aRows, bCols, aCols_bRows);
                }
            }
        }
    }
}
