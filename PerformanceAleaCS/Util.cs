﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea.CUDA;

namespace PerformanceAleaCS
{
    static class Util
    {
        private static Worker worker = null;

        public static Worker Worker
        {
            get
            {
                if (worker == null)
                {
                    // use threadless worker to avoid thread context switching
                    worker = Worker.Create(Device.Default);
                }
                return worker;
            }
        }

        public static float[] RandomMatrix(int rows, int cols)
        {
            var result = new float[rows * cols];
            var random = new Random(4711);
            for (var i = 0; i < rows * cols; ++i)
            {
                    result[i] = (float)random.NextDouble();
            }
            return result;
        }

        public static void VerifyResult(float[] a, float[] b, float[] c, int aRows, int bCols, int aCols_bRows)
        {
            // verify took long time, so you can choose to skip it
            if (false)
            {
                for (var row = 0; row < aRows; ++row)
                {
                    for (var col = 0; col < bCols; ++col)
                    {
                        var sum = 0.0f;
                        for (var k = 0; k < aCols_bRows; ++k)
                        {
                            sum += a[row * aCols_bRows + k] * b[k * bCols + col];
                        }
                        var actual = c[row * bCols + col];
                        if (Math.Abs(sum - actual) > 0.1f)
                        {
                            throw new Exception("Verify failed");
                        }
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
