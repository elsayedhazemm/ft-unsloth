import re
import asyncio
import aiohttp
from openai import AsyncOpenAI

# Grader selection: "gemma" or "gpt"
GRADER_MODE = "gpt"

GPT_GRADER = AsyncOpenAI(
    api_key="")

GRADING_SYSTEM_PROMPT = """You are an expert medical evaluator. Score how well an answer meets each criterion compared to a gold standard.
For each criterion, score 1-5:
1 = Missing/wrong
2 = Partially addressed with errors
3 = Addressed but incomplete
4 = Well addressed
5 = Fully and accurately addressed
Respond with ONLY comma-separated integers, one per criterion. Nothing else. Example for 3 criteria: 4,5,3"""

GRADING_USER_TEMPLATE = """
QUESTION:
{question}

GOLD ANSWER:
{gold}

CANDIDATE ANSWER:
{answer}

CRITERIA:
{criteria}

Scores (comma-separated, one per criterion):"""


def parse_criteria(assistant_content: str) -> tuple[str, list[str]]:
    """
    Parse gold answer and criteria from assistant message.
    """
    gold_match = re.search(r"<gold>(.*?)</gold>", assistant_content, re.DOTALL)
    gold = gold_match.group(1).strip() if gold_match else ""
    criteria = re.findall(r"<criteria>(.*?)</criteria>", assistant_content, re.DOTALL)
    criteria = [c.strip() for c in criteria]
    return gold, criteria


def parse_scores(text: str, expected_count: int) -> list[int]:
    """
    Parse comma-separated scores from grader response.
    """
    if not text:
        return []
    text = text.strip()
    numbers = re.findall(r"\b([1-5])\b", text)
    if len(numbers) >= expected_count:
        return [int(n) for n in numbers[:expected_count]]
    parts = [p.strip() for p in text.split(",")]
    scores = []
    for p in parts:
        try:
            score = int(p)
            if 1 <= score <= 5:
                scores.append(score)
        except ValueError:
            continue
    if len(scores) >= expected_count:
        return scores[:expected_count]
    return []


async def call_gpt_grader(
    question: str, gold: str, answer: str, criteria: list[str], max_retries: int = 3
) -> list[int]:
    """Call GPT grader."""
    if not answer or not criteria:
        return []

    criteria_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    prompt = GRADING_USER_TEMPLATE.format(
        gold=gold, question=question, answer=answer, criteria=criteria_text
    )
    print(prompt)
    for attempt in range(max_retries):
        try:
            response = await GPT_GRADER.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=256,
            )
            result_text = response.choices[0].message.content
            scores = parse_scores(result_text, len(criteria))
            if scores:
                return scores
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] GPT grading failed: {e}")
                return []
            await asyncio.sleep(0.5 * (attempt + 1))
    return []


def _compute(scores: list[int], num_criteria: int) -> float:
    """
    Compute final reward from scores.
    """
    if not scores or num_criteria == 0:
        return 0.0

    avg = sum(scores) / num_criteria
    r = avg / 5
    print(r)
    return r


async def grade_single(completion_content: str, assistant_content: str, question: str) -> float:
    """Grade a single completion and return reward."""
    answer = completion_content
    gold, criteria = parse_criteria(assistant_content)

    if not gold or not criteria:
        return -1.0

    scores = await call_gpt_grader(question, gold, answer, criteria)
    return _compute(scores, len(criteria))


async def accuracy_reward_async(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Async reward function for GRPO training.
    """
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    tasks = [grade_single(r, answer, question) for r in responses]
    rewards = await asyncio.gather(*tasks)

    return list(rewards)


def accuracy_reward(prompts, completions, answer, **kwargs):
    """
    Sync wrapper for async reward function.
    Main entry point for trainer.
    """
    return asyncio.run(accuracy_reward_async(prompts, completions, answer, **kwargs))
