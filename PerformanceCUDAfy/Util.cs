using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceCUDAfy
{
    static class Util
    {
        public static float[,] RandomMatrix(int rows, int cols)
        {
            var result = new float[rows, cols];
            var random = new Random(4711);
            for (var row = 0; row < rows; row++)
            {
                for (var col = 0; col < cols; col++)
                {
                    result[row, col] = (float)random.NextDouble();
                }
            }
            return result;
        }

        public static void VerifyResult(float[,] a, float[,] b, float[,] c)
        {
            for (var row = 0; row < c.GetLength(0); row++)
            {
                for (var col = 0; col < c.GetLength(1); col++)
                {
                    var sum = 0f;
                    for (var k = 0; k < a.GetLength(1); k++)
                    {
                        sum += a[row, k] * b[k, col];
                    }
                    if (Math.Abs(sum - c[row, col]) > 0.1)
                    {
                        throw new Exception();
                    }
                }
            }
        }

        public static int Divup(int a, int b)
        {
            return (a + b - 1) / b;
        }
    }
}
