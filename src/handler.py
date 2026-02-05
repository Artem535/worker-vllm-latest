import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = None
openai_engine = None

async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

if __name__ == "__main__":
    if vllm_engine is None:
        vllm_engine = vLLMEngine()

    if openai_engine is None:
        openai_engine = OpenAIvLLMEngine(vllm_engine)
    
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
            "return_aggregate_stream": True,
        }
    )
