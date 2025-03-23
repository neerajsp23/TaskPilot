from datetime import date
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = GroqModel("llama-3.3-70b-versatile")


class TaskCollectorSchema(BaseModel):
    task_name: str = Field(Description="Name of the task")
    description: str = Field(Description="Description of the task")
    assigned_date: date = Field(Description="Date when the task was assigned")

    def __str__(self):
        return f"Task: {self.task_name}\nDescription: {self.description}\nAssigned Date: {self.assigned_date}"


class TaskPrioritizerSchema(BaseModel):
    priority: int = Field(Description="Priority of the task")

    def __str__(self):
        return f"Task Priority: {self.priority}"


class TaskSummarizerSchema(BaseModel):
    task_name: str = Field(Description="Name of the task")
    description: str = Field(Description="Description of the task")
    summary: str = Field(Description="Summary of the tasks")
    priority: int = Field(Description="Priority of the task")
    insights: str = Field(Description="Insights of the task")
    tips: list[str] = Field(Description="Tips for the task")

    def __str__(self):
        return f"Task: {self.task_name}\nDescription: {self.description}\nSummary: {self.summary} \nPriority: {self.priority}\nInsights: {self.insights}\nTips: {', '.join(self.tips)}"


task_collector_agent = Agent(
    model=model,
    system_prompt="You are a task collector that understands the user's task and splits it into task_name, description, and assigned_date.",
    result_type=TaskCollectorSchema,
)

task_prioritizer_agent = Agent(
    model=model,
    system_prompt="You are a task prioritizer that understands the user's task and assigns the priority to the task in range of 1 to 10, 10 being the highest priority.",
    result_type=TaskPrioritizerSchema,
)

task_summarizer_agent = Agent(
    model=model,
    system_prompt="You are a task summarizer that understands the user's task and strictly develops a detailed summary of the task with description, priority,and insights.",
    result_type=TaskSummarizerSchema,
)

task_coordinator_agent = Agent(
    model=model,
    system_prompt="You are a task coordinator that understands the user's multiple tasks and time and the best tasigns .",
    result_type=TaskSummarizerSchema,
)

import asyncio


async def main():
    preprocessed_task = await task_collector_agent.run(
        "need to send an email to the client by tomorrow"
    )
    priority = await task_prioritizer_agent.run(str(preprocessed_task.data))
    print(str(preprocessed_task.data) + str(priority.data))
    summary = await task_summarizer_agent.run(
        str(preprocessed_task.data) + str(priority.data)
    )
    print(str(summary.data))


asyncio.run(main())
