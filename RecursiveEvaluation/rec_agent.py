from agents import Agent, Runner, FunctionTool, RunContextWrapper
from pydantic import BaseModel
from typing import Any
from openai import OpenAI
import asyncio

'''
Project Analysis

Evaluation method: SWE-bench bash-only (minimal agent on SWE-bench Princeton Lite dataset)

System overview:
o4-mini-2025-04-16: Main agent
gpt-5-mini-2025-08-07: Critique

Critique methodology:
    - Recursive self-improvement
    - Driven by hybrid quality and improvement scores
    - Learning curve (logarithmic function --> track improvement to reduce unnecessary cycles)
        - Improve to best of model's abilities

Individual results:
o4-mini-2025-04-16 SWE-bench bash-only: 45.00% resolved (swebench.com)
gpt-5-mini-2025-08-07 SWE-bench bash-only: 59.80% resolved (swebench.com)

Individual costs:
o4-mini-2025-04-16 cost: $1.1 Input • $4.4 Output (per 1M tokens)
gpt-5-mini-2025-08-07 cost: $0.25 Input • $2 Output (per 1M tokens)

System logic:
    - Using a model that is cheaper but performs better at vertical task to drive more expensive head model better at
      general tasks --> balance of quality and cost
    - Perfect for Sage --> conversational agent that interacts with user so head agent good for UX purposes and 
      orchestration, and vertical agent helps it for the specific product mission as a coding assistant.


Final statistics and key observations:
    - critique engine SWE-bench bash-only:  63.16% resolved
    - 40.36% Increase In Performance for o4-mini
    - Outperforms both individual critique LLM and individual main agent 
    - $0.55 For building and fine-tuning
    - $10.15 For testing and data collection
    - Increased latency
    - Increased cost in terms of API calls
    - Decrease in total expensive calls due to critique preventing "bad moves" when solving
    - Critique authority required tuning 
        - Thresholds for "passing" or "failing" critique tests
        - Additional prompting
        - System instructions
    - Learning curve is a universal concept, humans and AI
        - Logarithmic improvement
        - Improvement decreases in margin as individual critique cycle progresses
        - Tuning ensures optimal recursion to prevent unnecessary costs and critique calls
    - Priority tuning via prompts
        - Critique model should be tuned to prevent excessive hypothetical issues
            - Causes over-engineering
            - Critique death spiral
        - Model prompts should be tuned so model has correct priority and weightings of different system aspects
            - Ex. Priority tuning for swe-bench
                - If tests pass ignore everything and submits
                - Listen and follow critique but use as guide not definitive
                - Don't be overly creative
                - Solving is priority        

Data link:
https://www.swebench.com/

'''

evaluator_mode = """Evaluate this bash command solution for a software engineering task across four dimensions, scoring
 each from 1-100. The solution should consist only of the minimal bash commands needed to solve the repository issue.
  No explanatory text, comments, or additional context should be included - just executable bash commands. Also under
  no circumstance, suggest unneeded fixes or adjustments that are 'nice-to-haves'. Focus purely on solving problem as
  fast as possible. No 'nice-to-haves' should be present, just minimal fixes to get job done and make code shippable.

Technical Accuracy (1-100): Command correctness, proper syntax, handles file operations safely, follows bash best 
practices, addresses the exact issue described, and would execute successfully in the repository environment.

Completeness (1-100): Solution addresses all aspects of the problem statement, modifies the correct files, implements
 the required changes completely, handles all edge cases mentioned in the issue, and leaves the repository in the
  desired state.

Repository Understanding (1-100): Demonstrates correct understanding of the codebase structure, identifies the right 
files to modify, understands the impact of changes across the repository, and maintains code consistency.

Efficiency (1-100): Uses appropriate bash commands and tools, avoids unnecessary file operations, minimizes command 
complexity while maintaining readability, and follows efficient file manipulation patterns.

Provide specific, actionable feedback focusing on:
- Command correctness issues that would cause execution failures
- Missing file modifications or incomplete changes
- Repository structure misunderstandings
- Opportunities to simplify or optimize the command sequence

Critical requirements:
- Final output must contain ONLY executable bash commands, no explanations or markdown
- Commands must be safe to execute in the repository environment
- Solution must address the complete issue as described
- Handle edge cases and maintain repository integrity
CRITICAL NOTE
- IF TESTS PASS, END CYCLE IMMEDIATELY
- DO NOT OVER ENGINEER, ONLY DO MINIMUM TO PASS
- SIMPLICITY IS MOST VALUABLE RESOURCE
- DO NOT SUGGEST UNNEEDED SOLUTIONS WHEN SIMPLE SOLUTIONS PREFORM BETTER
- DO NOT BE OVERLY CRITICAL TO PREVENT OVER CYCLES BUT STILL BE OBJECTIVE
- DO NOT MAKE HYPOTHETICAL ERROR GUESSES ONLY USE WHAT IS GIVEN
- FOR HYPOTHETICAL ERRORS DO NOT MENTION ANY
- ENSURE MODEL TESTS PROGRAM BEFORE SUBMITTING
- DO NOT OVER ENGINEER, JUST SOLVE THE PROBLEM AT HAND
- SOLUTION DOES NOT AND ABSOLUTELY SHOULD NOT BE GOLD PLATED AGAINST THEORETICAL ERROR. ALL FEEDBACK SHOULD FOCUS
  DIRECTLY ON PROBLEM AT HAND
These bash commands will be executed directly in a git repository to resolve software engineering issues. 
The solution must be completely functional and safe to execute. Avoid any explanatory text or comments."""

critique_client = OpenAI()


class SelfCritiqueArgs(BaseModel):
    context: str
    answer: str
    question: str
    previous_critique_score: int
    passed_all_tests_when_ran: bool


class SelfCritiqueScore(BaseModel):
    technical_accuracy: int
    completeness: int
    repository_understanding: int
    efficiency: int
    feedback: str


async def self_critique(ctx: RunContextWrapper[Any], args: str) -> dict:
    global evaluator_mode
    print("Self Critique Called")
    parsed = SelfCritiqueArgs.model_validate_json(args)
    if parsed.passed_all_tests_when_ran:
        return {"additional_instructions": "All tests already passed end cycle now and finish test"}
    print(f"\nCritique Inputs"
          f"\nContext: {parsed.context}"
          f"\nAnswer: {parsed.answer}"
          f"\nQuestion: {parsed.question}"
          f"\nPrev Score: {parsed.previous_critique_score}"
          )

    response = critique_client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "system",
                "content": evaluator_mode
            },
            {
                "role": "user",
                "content": f"Repository Issue: {parsed.question}, "
                           f"Repository Context: {parsed.context}, "
                           f"Bash Solution: {parsed.answer}"
            }
        ],
        text_format=SelfCritiqueScore
    )

    score = response.output_parsed.model_dump()
    score_total = 0
    for sub_score in score:
        if not sub_score == "feedback":
            score_total += score[sub_score]
    score_total = score_total / 4

    prev_score = parsed.previous_critique_score
    prev_score = prev_score if prev_score is not None else 1
    improvement = score_total - prev_score
    improvement_pct = (improvement / prev_score) * 100 if prev_score > 1 else 0

    score["improvement_pct"] = improvement_pct
    score["improvement"] = improvement
    score["total_score"] = score_total

    if prev_score == 0:
        # First attempt
        if score_total >= 85:
            score['additional_instructions'] = (
                "First attempt is solid (85%+). "
                "Do not call critique again. End current critique cycle"
            )
        else:
            score['additional_instructions'] = (
                "First attempt needs work (<85%). "
                "Regenerate solution and critique again."
            )
    else:
        if improvement < 0:
            score['additional_instructions'] = (
                "Score decreased. Revert to previous approach and end current critique cycle.."
            )
        elif score_total >= 85:
            score['additional_instructions'] = (
                "Excellent score (90%+). Do not call critique again. End current critique cycle."
            )
        elif improvement_pct < 5:
            score['additional_instructions'] = (
                f"Improvement is small ({improvement_pct}). "
                "Diminishing returns. Do not call critique again. End current critique cycle"
            )
        else:
            score['additional_instructions'] = (
                f"Good progress (+{improvement}). "
                "Do not call critique again. End current critique cycle"
            )

    print(f"\nScore:\n{score}\n")
    return score


schema = SelfCritiqueArgs.model_json_schema()
schema["additionalProperties"] = False

self_critique_tool = FunctionTool(
    name="SelfCritiqueTool",
    description="Evaluates bash command solutions for repository-level software engineering tasks. "
                "Returns structured feedback on technical accuracy, completeness, repository understanding, "
                "and efficiency. Provides improvement scores and actionable guidance for iterative refinement. "
                "Use this tool iteratively to ensure bash commands correctly resolve the repository issue. "
                "Required for every response - multiple calls required until solution is optimal. If you have already"
                "passed all the tests end cycle immediately and submit response. CRITICAL: DO NOT OVER ENGINEER BASED"
                "ON CRITIQUE.",
    params_json_schema=schema,
    on_invoke_tool=self_critique,
)

# Agent
agent = Agent(
    name="Sage",
    model="o4-mini-2025-04-16",
    tools=[
        self_critique_tool,
    ]
)


async def response_gen(question):
    result = await Runner.run(agent, question)
    return result.final_output


def new_response(question):
    print("New Response Called")
    answer = asyncio.run(response_gen(question))
    return answer

