import re
import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

# ============================================================================
# FORMAT TAGS - must match the system prompt in qadataset.py
# ============================================================================
THINKING_START = "<think>"
THINKING_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# ============================================================================
# GRADER CONFIGURATION
# ============================================================================
GPT_GRADER = AsyncOpenAI(api_key="")

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

# ============================================================================
# GENERATION LOGGER
# ============================================================================
class GenerationLogger:
    """Thread-safe logger for saving all generations and scores to JSONL."""

    def __init__(self, log_dir: str = "outputs/generations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"generations_{timestamp}.jsonl"
        self._lock = threading.Lock()
        self._step = 0

    def log(
        self,
        question: str,
        generation: str,
        extracted_answer: str | None,
        gold_answer: str,
        criteria: list[str],
        llm_scores: list[int],
        accuracy_reward: float,
        format_exact_reward: float,
        format_approx_reward: float,
        total_reward: float,
    ):
        """Log a single generation with all its scores."""
        record = {
            "step": self._step,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "generation": generation,
            "extracted_answer": extracted_answer,
            "gold_answer": gold_answer,
            "criteria": criteria,
            "llm_scores": llm_scores,
            "rewards": {
                "accuracy": accuracy_reward,
                "format_exact": format_exact_reward,
                "format_approx": format_approx_reward,
                "total": total_reward,
            },
        }
        with self._lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    def increment_step(self):
        """Call this at the start of each training step."""
        with self._lock:
            self._step += 1


# Global logger instance
GENERATION_LOGGER = GenerationLogger()


# ============================================================================
# FORMAT MATCHING REGEX
# ============================================================================
# Matches the exact expected format: <think>...</think>...<answer>...</answer>
match_format = re.compile(
    rf"^\s*"
    rf"{re.escape(THINKING_START)}.+?{re.escape(THINKING_END)}.*?"
    rf"{re.escape(ANSWER_START)}(.+?){re.escape(ANSWER_END)}"
    rf"\s*$",
    flags=re.MULTILINE | re.DOTALL
)

# Extract just the answer content
extract_answer = re.compile(
    rf"{re.escape(ANSWER_START)}(.+?){re.escape(ANSWER_END)}",
    flags=re.MULTILINE | re.DOTALL
)


# ============================================================================
# FORMAT REWARD FUNCTIONS
# ============================================================================
def compute_format_exact(response: str) -> float:
    """
    Reward for matching the exact format: <think>...</think><answer>...</answer>
    Returns 1.0 if format matches exactly, 0.0 otherwise.
    """
    if match_format.search(response) is not None:
        return 1.0
    return 0.0


def compute_format_approx(response: str) -> float:
    """
    Partial credit for including format elements.
    Each tag appearing exactly once: +0.25
    Each tag appearing 0 or >1 times: -0.25

    Max score: +1.0, Min score: -1.0
    """
    score = 0.0
    score += 0.25 if response.count(THINKING_START) == 1 else -0.25
    score += 0.25 if response.count(THINKING_END) == 1 else -0.25
    score += 0.25 if response.count(ANSWER_START) == 1 else -0.25
    score += 0.25 if response.count(ANSWER_END) == 1 else -0.25
    return score


def match_format_exactly(completions, **_kwargs) -> list[float]:
    """
    Reward function for exact format matching.
    Returns 1.0 if format matches exactly, 0.0 otherwise.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(compute_format_exact(response))
    return scores


def match_format_approximately(completions, **_kwargs) -> list[float]:
    """
    Reward function for approximate format matching.
    Max score: +1.0, Min score: -1.0
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(compute_format_approx(response))
    return scores


# ============================================================================
# PARSING UTILITIES
# ============================================================================
def parse_criteria(assistant_content: str) -> tuple[str, list[str]]:
    """
    Parse gold answer and criteria from assistant message in the dataset.
    Expected format in dataset:
    <gold>...</gold>
    <criteria>...</criteria>
    <criteria>...</criteria>
    ...
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


def extract_answer_content(response: str) -> str | None:
    """
    Extract the content between <answer> and </answer> tags.
    Returns None if tags not found.
    """
    match = extract_answer.search(response)
    if match:
        return match.group(1).strip()
    return None


# ============================================================================
# LLM GRADING
# ============================================================================
async def call_gpt_grader(
    question: str, gold: str, answer: str, criteria: list[str], max_retries: int = 3
) -> list[int]:
    """Call GPT grader to evaluate answer against criteria."""
    if not answer or not criteria:
        return []

    criteria_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    prompt = GRADING_USER_TEMPLATE.format(
        gold=gold, question=question, answer=answer, criteria=criteria_text
    )

    for attempt in range(max_retries):
        try:
            response = await GPT_GRADER.chat.completions.create(
                model="gpt-4o-mini",
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


def compute_accuracy_reward(scores: list[int], num_criteria: int) -> float:
    """
    Compute accuracy reward from LLM grader scores.

    Returns a value between 0.0 and 3.0 to make correctness the dominant signal.
    - Average score 1 (worst) -> 0.0
    - Average score 5 (best)  -> 3.0
    """
    if not scores or num_criteria == 0:
        return 0.0

    avg = sum(scores) / num_criteria  # 1.0 to 5.0
    # Normalize to 0-1 range, then scale to 0-3
    normalized = (avg - 1) / 4  # 0.0 to 1.0
    return normalized * 3.0  # 0.0 to 3.0


# ============================================================================
# GRADING RESULT DATACLASS
# ============================================================================
class GradeResult:
    """Container for grading results to facilitate logging."""

    def __init__(
        self,
        generation: str,
        extracted_answer: str | None,
        gold: str,
        criteria: list[str],
        llm_scores: list[int],
        accuracy_reward: float,
        format_exact: float,
        format_approx: float,
    ):
        self.generation = generation
        self.extracted_answer = extracted_answer
        self.gold = gold
        self.criteria = criteria
        self.llm_scores = llm_scores
        self.accuracy_reward = accuracy_reward
        self.format_exact = format_exact
        self.format_approx = format_approx
        self.total_reward = accuracy_reward + format_exact + format_approx


async def grade_single(
    completion_content: str, assistant_content: str, question: str
) -> GradeResult:
    """
    Grade a single completion using LLM judge.

    Extracts the answer from <answer> tags before grading.
    Returns GradeResult with all scoring information.
    """
    # Extract answer from tags
    extracted = extract_answer_content(completion_content)
    answer_to_grade = extracted if extracted else completion_content

    # Parse gold and criteria from dataset
    gold, criteria = parse_criteria(assistant_content)

    # Compute format rewards
    format_exact = compute_format_exact(completion_content)
    format_approx = compute_format_approx(completion_content)

    if not gold or not criteria:
        print("[WARNING] Missing gold or criteria in dataset")
        return GradeResult(
            generation=completion_content,
            extracted_answer=extracted,
            gold=gold,
            criteria=criteria,
            llm_scores=[],
            accuracy_reward=0.0,
            format_exact=format_exact,
            format_approx=format_approx,
        )

    # Get LLM scores
    llm_scores = await call_gpt_grader(question, gold, answer_to_grade, criteria)
    accuracy = compute_accuracy_reward(llm_scores, len(criteria))

    result = GradeResult(
        generation=completion_content,
        extracted_answer=extracted,
        gold=gold,
        criteria=criteria,
        llm_scores=llm_scores,
        accuracy_reward=accuracy,
        format_exact=format_exact,
        format_approx=format_approx,
    )

    print(
        f"[GRADE] LLM: {llm_scores} -> Acc: {accuracy:.2f} | "
        f"Fmt: exact={format_exact:.1f}, approx={format_approx:.2f} | "
        f"Total: {result.total_reward:.2f}"
    )

    return result


async def accuracy_reward_async(prompts, completions, answer, **_kwargs) -> list[float]:
    """
    Async reward function for LLM-based grading.

    Returns rewards scaled 0.0 to 3.0 (correctness is the main signal).
    Also logs all generations and scores to file.
    """
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    # Grade all completions
    tasks = [grade_single(r, answer, question) for r in responses]
    results: list[GradeResult] = await asyncio.gather(*tasks)

    # Increment step counter
    GENERATION_LOGGER.increment_step()

    # Log each generation
    for result in results:
        GENERATION_LOGGER.log(
            question=question,
            generation=result.generation,
            extracted_answer=result.extracted_answer,
            gold_answer=result.gold,
            criteria=result.criteria,
            llm_scores=result.llm_scores,
            accuracy_reward=result.accuracy_reward,
            format_exact_reward=result.format_exact,
            format_approx_reward=result.format_approx,
            total_reward=result.total_reward,
        )

    # Return only accuracy rewards (format rewards handled by separate functions)
    return [r.accuracy_reward for r in results]


def accuracy_reward(prompts, completions, answer, **_kwargs) -> list[float]:
    """
    Sync wrapper for async reward function.
    Main entry point for trainer - evaluates correctness via LLM judge.

    Reward range: 0.0 to 3.0

    Note: All generations are logged to outputs/generations/generations_<timestamp>.jsonl
    """
    return asyncio.run(accuracy_reward_async(prompts, completions, answer))
