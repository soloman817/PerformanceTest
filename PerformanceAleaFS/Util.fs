module Util

open System

let randomMatrix (rows:int) (cols:int) =
    let rng = Random(4711)
    Array.init (rows * cols) (fun _ -> rng.NextDouble() |> float32)

let verifyResult (a:float32[]) (b:float32[]) (c:float32[]) (aRows:int) (bCols:int) (aCols_bRows:int) =
    // verify took long time, so you can choose to skip it
    if false then
        for row = 0 to aRows - 1 do
            for col = 0 to bCols - 1 do
                let mutable sum = 0.0f
                for k = 0 to aCols_bRows - 1 do
                    sum <- sum + a.[row * aCols_bRows + k] * b.[k * bCols + col]
                let sum' = c.[row * bCols + col]
                if (abs (sum - sum')) > 0.1f then
                    failwith "Verify failed"