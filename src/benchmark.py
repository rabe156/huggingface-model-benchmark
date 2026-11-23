from .inference import predict
import time
import numpy as np
import pandas as pd

def benchmark_model(dataset,tokenizer,model,batch_sizes=(1, 4, 8, 16),repetitions=3):
    benchmark_results=[]
    for batch_size in batch_sizes :
        warmup_dataset=dataset.select(range(min(batch_size*2,len(dataset))))
        _,c=predict(warmup_dataset,tokenizer,model,batch_size)
        times=[]
        for repetition in range(repetitions) : 
           start=time.perf_counter() 
           label,confidence=predict(dataset,tokenizer,model,batch_size)
           end=time.perf_counter()
           elapsed=end-start
           times.append(elapsed)
        execution_time=np.median(times)
        throughput=len(dataset)/execution_time
        time_per_sample=(1/throughput)*1000
        benchmark = {
            "batch_size": batch_size,
            "median_time_s": execution_time,
            "throughput": throughput,
            "ms_per_sample": time_per_sample
        }

        benchmark_results.append(benchmark)

    return pd.DataFrame(benchmark_results)


