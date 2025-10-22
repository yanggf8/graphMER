import asyncio
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    priority: int = 1

async def process_tasks(tasks: List[Task]) -> None:
    for task in tasks:
        await execute_task(task)
        
async def execute_task(task: Task) -> bool:
    print(f"Processing {task.name}")
    await asyncio.sleep(0.1)
    return True

def create_task(name: str) -> Task:
    return Task(name=name, priority=1)
