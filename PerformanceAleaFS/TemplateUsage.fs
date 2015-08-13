module TemplateUsage

open System.Diagnostics
open Alea.CUDA
open Alea.CUDA.Utilities

let templateSimpleMultiply = cuda {

    let tileSize = 32
    
    let! kernel =
        <@ fun (a:deviceptr<float32>) (b:deviceptr<float32>) (c:deviceptr<float32>)
               (aRows:int) (bCols:int) (aCols_bRows:int) ->
            let row = blockDim.y * blockIdx.y + threadIdx.y
            let col = blockDim.x * blockIdx.x + threadIdx.x
            if (row < aRows && col < bCols) then
                let mutable sum = 0.0f
                for k = 0 to aCols_bRows - 1 do
                    sum <- sum + a.[row * aCols_bRows + k] * b.[k * bCols + col]
                c.[row * bCols + col] <- sum
        @> |> Compiler.DefineKernel

    return Entry(fun program ->
        let worker = program.Worker
        let kernel = program.Apply kernel

        let run () =
            let factor = 8
            let a = Util.randomMatrix (100*factor) (200*factor)
            let b = Util.randomMatrix (200*factor) (300*factor)
            let aRows = 100*factor
            let bCols = 300*factor
            let aCols_bRows = 200*factor
            use da = worker.Malloc(a)
            use db = worker.Malloc(b)
            use dc = worker.Malloc<float32>(aRows * bCols)
            let lp = LaunchParam(dim3(divup bCols tileSize, divup aRows tileSize), dim3(tileSize, tileSize))
            
            let timer = Stopwatch.StartNew()
            kernel.Launch lp da.Ptr db.Ptr dc.Ptr aRows bCols aCols_bRows
            worker.Synchronize()
            timer.Stop()
            printfn "Kernel launch first time               %f ms" timer.Elapsed.TotalMilliseconds

            let repetitions = 50
            timer.Restart()
            for i = 1 to repetitions do
                kernel.Launch lp da.Ptr db.Ptr dc.Ptr aRows bCols aCols_bRows
            worker.Synchronize()
            timer.Stop()
            printfn "Kernel launch average time             %f ms" (timer.Elapsed.TotalMilliseconds / (float repetitions))
                       
            let c = dc.Gather()
            Util.verifyResult a b c aRows bCols aCols_bRows 

        run ) }

let testSimpleMutiply() =
    for iter = 1 to 3 do
        printfn "====> Test SimpleMultiply with Alea GPU F# template usage (#.%d) <====" iter

        let timer = Stopwatch.StartNew()
        let worker = Worker.Default
        printfn "GPU: %s" worker.Device.Name
        timer.Stop()
        printfn "Step 1) Runtime setup                  %f ms" timer.Elapsed.TotalMilliseconds

        timer.Restart()
        let irm = Compiler.Compile(templateSimpleMultiply, worker.DefaultCompileOptions).IRModule
        let ptxm = Compiler.Link(irm).PTXModule
        timer.Stop()
        printfn "Step 2) Compile                        %f ms" timer.Elapsed.TotalMilliseconds

        timer.Restart()
        use program = worker.LoadProgram(ptxm)
        timer.Stop()
        printfn "Step 3) Load module                    %f ms" timer.Elapsed.TotalMilliseconds

        program.Run()

    