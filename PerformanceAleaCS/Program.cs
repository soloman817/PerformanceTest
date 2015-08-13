using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerformanceAleaCS
{
    class Program
    {
        static void Main(string[] args)
        {
            InstanceUsageJIT.TestSimpleMultiply();
            InstanceUsageAOT.TestSimpleMultiply();
            MethodUsageJIT.TestSimpleMultiply();
            MethodUsageAOT.TestSimpleMultiply();
        }
    }
}
