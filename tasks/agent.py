from datetime import date, time
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

model = GroqModel("llama-3.3-70b-versatile")


# MARK: Schema
class TaskCollectorSchema(BaseModel):
    task_name: str = Field(Description="Name of the task")
    description: str = Field(Description="Description of the task")
    assigned_date: date = Field(Description="Date when the task was assigned")
    due_date: Optional[date] = Field(default=None, Description="Due date of the task")

    def __str__(self):
        return (
            f"Task: {self.task_name}\nDescription: {self.description}\nAssigned Date: {self.assigned_date}\nDue Date: {self.due_date}"
            if self.due_date
            else f"Task: {self.task_name}\nDescription: {self.description}\nAssigned Date: {self.assigned_date}"
        )


class TaskPrioritizerSchema(BaseModel):
    priority: int = Field(Description="Priority of the task")

    def __str__(self):
        return f"Task Priority: {self.priority}"


class TaskSummarizerSchema(BaseModel):
    summary: str = Field(Description="Summary of the tasks")
    insights: str = Field(Description="Insights of the task")
    subtasks: list[str] = Field(Description="Subtasks of the task")
    tips: list[str] = Field(Description="Tips for the task")

    def __str__(self):
        return f"Task: {self.task_name}\nSummary: {self.summary} \nInsights: {self.insights}\nSubtasks: {self.subtasks}\nTips: {', '.join(self.tips)}"


class SubtaskSchema(BaseModel):
    subtask_name: str = Field(description="Name of the subtask")
    start_time: time = Field(description="Start time of the subtask")
    end_time: time = Field(description="End time of the subtask")

    @property
    def duration(self) -> int:
        """Returns the duration in minutes."""
        start_minutes = self.start_time.hour * 60 + self.start_time.minute
        end_minutes = self.end_time.hour * 60 + self.end_time.minute
        return end_minutes - start_minutes

    def __str__(self):
        return f"Subtask: {self.subtask_name} | Start: {self.start_time} | End: {self.end_time}"


class TaskCoordinatorSchema(BaseModel):
    start_time: time = Field(description="Start time of the project")
    end_time: time = Field(description="End time of the project")
    subtasks: Optional[List[SubtaskSchema]] = Field(
        default=[], description="List of subtasks with time durations"
    )

    @property
    def total_project_duration(self) -> int:
        start_minutes = self.start_time.hour * 60 + self.start_time.minute
        end_minutes = self.end_time.hour * 60 + self.end_time.minute
        return end_minutes - start_minutes

    @property
    def suggested_subtask_durations(self) -> List[str]:
        total_subtask_minutes = sum(subtask.duration for subtask in self.subtasks)
        remaining_minutes = self.total_project_duration - total_subtask_minutes
        suggestions = []

        if remaining_minutes > 0 and self.subtasks:
            per_subtask = remaining_minutes // len(self.subtasks)
            for subtask in self.subtasks:
                suggestions.append(
                    f"Suggested for {subtask.subtask_name}: {subtask.duration + per_subtask} minutes"
                )
        else:
            suggestions = [
                f"Subtask '{subtask.subtask_name}' already occupies {subtask.duration} minutes."
                for subtask in self.subtasks
            ]

        return suggestions

    @model_validator(mode="before")
    def validate_times(cls, values):
        start_time = values.get("start_time")
        end_time = values.get("end_time")
        if end_time <= start_time:
            raise ValueError("End time must be after the start time.")
        return values

    def __str__(self):
        subtasks_str = (
            "\n  ".join(str(subtask) for subtask in self.subtasks)
            if self.subtasks
            else "No Subtasks"
        )
        return (
            f"Start Time: {self.start_time}\n"
            f"End Time: {self.end_time}\n"
            f"Subtasks:\n  {subtasks_str}"
        )


# MARK: Agents
class TaskAgents:
    def __init__(self):
        self.task_collector_agent = Agent(
            model=model,
            system_prompt="You are a task collector that understands the user's task and splits it into task_name, description, and assigned_date.",
            result_type=TaskCollectorSchema,
        )

        self.task_prioritizer_agent = Agent(
            model=model,
            system_prompt="You are a task prioritizer that understands the user's task and assigns the priority to the task in range of 1 to 10, 10 being the highest priority.",
            result_type=TaskPrioritizerSchema,
        )

        self.task_summarizer_agent = Agent(
            model=model,
            system_prompt="You are a task summarizer that understands the user's task and strictly develops a detailed summary of the task with description, priority,and insights.",
            result_type=TaskSummarizerSchema,
        )

        self.task_coordinator_agent = Agent(
            model=model,
            system_prompt=(
                "You are a task coordinator that understands the user's multiple tasks and time and the earliest time and minimal possible duration for the task .If its small task make sure you assign the minimal time and dont assign already allocated time strictly",
                "use the `occupied_time` tool to get the occupied time",
            ),
            result_type=TaskCoordinatorSchema,
        )

        @self.task_coordinator_agent.tool_plain
        async def occupied_time():
            # return [ {"task_name":"breakfast","start_time":time(9,0),"end_time":time(9,30)},{"task_name":"build restapi","start_time":time(9,0),"end_time":time(13,30)} ]
            return "breakfast : 9:00-9:30, build restapi : 9:00-13:30"


from typing import TypedDict
from datetime import date, time
from langgraph.graph import StateGraph, START, END
from pydantic_ai import Agent
import asyncpg
import json

DB_CONFIG = {
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase",
    "host": "localhost",
    "port": 5432,
}


# MARK: Graph State Schema
class TaskAssignState(TypedDict):
    user_input: str
    task_name: str
    description: str
    assigned_date: date
    due_date: date
    priority: int
    summary: str
    insights: str
    subtasks: list[str]
    subtasks_duration: Optional[List[SubtaskSchema]]
    tips: list[str]
    start_time: time
    end_time: time


# MARK: Graph
class TaskGraph(TaskAgents):

    def __init__(self):
        super().__init__()
        builder = StateGraph(TaskAssignState)
        self.graph = self.setup_graph(builder)

    def setup_graph(self, builder):
        builder.add_node("task_collector", self.task_collector)
        builder.add_node("task_prioritizer", self.task_prioritizer)
        builder.add_node("task_summarizer", self.task_summarizer)
        builder.add_node("task_coordinator", self.task_coordinator)

        builder.add_edge(START, "task_collector")
        builder.add_edge("task_collector", "task_prioritizer")
        builder.add_edge("task_prioritizer", "task_summarizer")
        builder.add_edge("task_summarizer", "task_coordinator")
        builder.add_edge("task_coordinator", END)

        graph = builder.compile()
        return graph

    async def task_collector(self, state: TaskAssignState):
        task_collection = await self.task_collector_agent.run(state["user_input"])
        state["task_name"] = task_collection.data.task_name
        state["description"] = task_collection.data.description
        state["assigned_date"] = task_collection.data.assigned_date
        state["due_date"] = task_collection.data.due_date
        return state

    async def task_prioritizer(self, state: TaskAssignState):
        input_prompt = f"Task Name: {state['task_name']}\nDescription: {state['description']}\nAssigned Date: {state['assigned_date']}\nDue Date: {state['due_date']}"
        task_priority = await self.task_prioritizer_agent.run(input_prompt)
        state["priority"] = task_priority.data.priority
        return state

    async def task_summarizer(self, state: TaskAssignState):
        input_prompt = f"Task Name: {state['task_name']}\nDescription: {state['description']}\nAssigned Date: {state['assigned_date']}\nDue Date: {state['due_date']} \nPriority: {state['priority']}"
        task_summary = await self.task_summarizer_agent.run(input_prompt)
        state["summary"] = task_summary.data.summary
        state["insights"] = task_summary.data.insights
        state["subtasks"] = task_summary.data.subtasks
        state["tips"] = task_summary.data.tips
        return state

    async def task_coordinator(self, state: TaskAssignState):
        input_prompt = f"Task Name: {state['task_name']}\nAssigned Date: {state['assigned_date']}\nDue Date: {state['due_date']} \nPriority: {state['priority']} \nSubtasks: {state['subtasks']} "
        task_coordinator = await self.task_coordinator_agent.run(input_prompt)
        state["start_time"] = task_coordinator.data.start_time
        state["end_time"] = task_coordinator.data.end_time
        state["subtasks_duration"] = task_coordinator.data.subtasks
        return state

    async def insert_last_event(self, last_event: dict):
        conn = await asyncpg.connect(**DB_CONFIG)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                user_input TEXT,
                task_name TEXT,
                description TEXT,
                assigned_date DATE,
                due_date DATE,
                priority INT,
                summary TEXT,
                insights TEXT,
                subtasks TEXT[],
                tips TEXT[],
                start_time TIME,
                end_time TIME,
                subtasks_duration JSONB  -- ✅ Store structured data as JSONB
            );
        """
        )

        await conn.execute(
            """
            INSERT INTO tasks (
                user_input, task_name, description, assigned_date, due_date, priority,
                summary, insights, subtasks, tips, start_time, end_time, subtasks_duration
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            );
        """,
            last_event["user_input"],
            last_event["task_name"],
            last_event["description"],
            last_event["assigned_date"],
            last_event["due_date"],
            last_event["priority"],
            last_event["summary"],
            last_event["insights"],
            last_event["subtasks"],
            last_event["tips"],
            last_event["start_time"],
            last_event["end_time"],
            json.dumps(
                [
                    {
                        "subtask_name": subtask.subtask_name,
                        "start_time": subtask.start_time.strftime("%H:%M:%S"),
                        "end_time": subtask.end_time.strftime("%H:%M:%S"),
                    }
                    for subtask in last_event["subtasks_duration"]
                ]
            ),  # ✅ Convert to JSON
        )

    async def stream_graph_updates(self, user_input: str):
        # async for event in self.graph.astream({"user_input": user_input}):
        #     print("event")
        #     for value in event.values():
        #         print("Assistant:", value)
        events = [
            event async for event in self.graph.astream({"user_input": user_input})
        ]

        if events:  # Ensure there is at least one event
            last_event = events[-1]
            print("Last Event:")
            for value in last_event.values():
                print("Assistant:", value)
                await self.insert_last_event(value)


import asyncio


async def main():
    task_graph = TaskGraph()
    user_input = "Need to send an email to Robin by evening"
    # user_input = "I have a meeting with the client tomorrow at 10:00 AM and I need to prepare a presentation and send it to them by 9:00 AM."
    await task_graph.stream_graph_updates(user_input)


asyncio.run(main())
