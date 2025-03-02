# How to Use LiveBench with BrainChimera

This guide explains how to use LiveBench to evaluate your BrainChimera model's performance across different capabilities. LiveBench is a contamination-free benchmark that provides objective evaluation in six key areas: math, coding, reasoning, language, instruction following, and data analysis.

## Overview

LiveBench provides several advantages for evaluating BrainChimera's performance:

- **Contamination-Free**: Questions are based on recent information to avoid test set contamination
- **Objective Evaluation**: Uses ground truth answers rather than LLM judges
- **Diverse Categories**: Tests multiple capabilities to identify strengths and weaknesses

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   cd livebench
   pip install anthropic openai pandas datasets
   pip install git+https://github.com/lm-sys/FastChat.git
   ```

2. **Download Questions**:
   ```bash
   cd livebench
   python -m livebench.download_questions
   ```
   
   If the download fails, you can create a sample set of questions:
   ```bash
   cd livebench
   python sample_livebench_questions.py
   python fix_json_format.py
   ```

## Evaluation Options

### Option 1: Quick Evaluation with Helper Function

The simplest way to evaluate your model is to use the `evaluate_with_livebench` function which was added to the COCONUTWLatentThinking.py file:

```python
from COCONUTWLatentThinking import evaluate_with_livebench

# Evaluate your model with default settings
results, metrics = evaluate_with_livebench(your_model)

# Or specify custom paths
results, metrics = evaluate_with_livebench(
    model=your_model,
    questions_file="livebench/brain_chimera_livebench_fixed.jsonl",
    output_path="livebench/custom_results.json"
)
```

### Option 2: Using the LiveBenchEvaluator Class

For more control over the evaluation process, you can use the `LiveBenchEvaluator` class directly:

```python
from COCONUTWLatentThinking import LiveBenchEvaluator

# Initialize the evaluator
evaluator = LiveBenchEvaluator(
    model=your_model,
    questions_file="livebench/brain_chimera_livebench_fixed.jsonl",
    output_path="livebench/brain_chimera_results.json"
)

# Run the evaluation
results, metrics = evaluator.run_full_evaluation()
```

## Understanding Results

The evaluation results contain:

1. **Overall Score**: The average score across all categories
2. **Category Scores**: Performance in each of the six categories
3. **Task Scores**: Detailed performance on specific tasks within each category
4. **Processing Time**: Average time taken to process questions

A higher score indicates better performance in that category or task.

## Example Output

```
===== LIVEBENCH EVALUATION RESULTS =====
Overall score: 0.4285 (120 questions)
Average processing time: 3.75 seconds per question

--- BY CATEGORY ---
coding: 0.5250 (20 questions, 4.21s avg)
data_analysis: 0.3750 (20 questions, 5.35s avg)
instruction_following: 0.6500 (20 questions, 2.87s avg)
language: 0.3250 (20 questions, 3.12s avg)
math: 0.4000 (20 questions, 4.53s avg)
reasoning: 0.2960 (20 questions, 2.45s avg)

--- BY TASK ---
AMPS_Hard: 0.4545 (11 questions, 5.12s avg)
LCB_generation: 0.6000 (10 questions, 3.78s avg)
coding_completion: 0.4500 (10 questions, 4.63s avg)
...
```

## Interpreting Performance

- **High Scores (>0.7)**: Excellent performance in that category
- **Medium Scores (0.4-0.7)**: Acceptable performance, but room for improvement
- **Low Scores (<0.4)**: Areas that need significant improvement

Look for patterns in the results to guide your model training:
- If math scores are low but language scores are high, focus more on math training
- If certain tasks within a category perform worse than others, target those specific tasks
- Compare performance across categories to identify relative strengths and weaknesses

## Customizing Evaluation

You can evaluate specific subsets of LiveBench by creating custom question files:

```python
# Evaluate only math questions
evaluator = LiveBenchEvaluator(
    model=your_model,
    questions_file="livebench/custom_math_only.jsonl",
    output_path="livebench/math_results.json"
)
results, metrics = evaluator.run_full_evaluation()
```

## Troubleshooting

If you encounter issues:

1. **Model Interface Error**: Ensure your model exposes a compatible interface (forward, generate, or generate_text method)
2. **JSON Format Issues**: Run the fix_json_format.py script on your questions file
3. **Missing Dependencies**: Verify all required packages are installed
4. **Memory Issues**: Reduce batch size or evaluate fewer questions at once

## Conclusion

LiveBench provides an objective, contamination-free way to evaluate your BrainChimera model's performance across multiple capabilities. Use the results to guide your training and improvement efforts, focusing on categories and tasks where the model performs poorly.

By regularly evaluating with LiveBench as you develop your model, you can track improvements and ensure that your training methods are working as expected.

## New Livebench Set-up
# How to Use LiveBench with BrainChimera

This guide explains how to use LiveBench to evaluate your BrainChimera model's performance across different capabilities. LiveBench is a contamination-free benchmark that provides objective evaluation in six key areas: math, coding, reasoning, language, instruction following, and data analysis.

## Overview

LiveBench provides several advantages for evaluating BrainChimera's performance:

- **Contamination-Free**: Questions are based on recent information to avoid test set contamination
- **Objective Evaluation**: Uses ground truth answers rather than LLM judges
- **Diverse Categories**: Tests multiple capabilities to identify strengths and weaknesses

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   cd livebench
   pip install anthropic openai pandas datasets
   pip install git+https://github.com/lm-sys/FastChat.git
Download Questions:

cd livebench
python -m livebench.download_questions
If the download fails, you can create a sample set of questions:

cd livebench
python sample_livebench_questions.py
python fix_json_format.py
Evaluation Options
Option 1: Quick Evaluation with Helper Function
The simplest way to evaluate your model is to use the evaluate_with_livebench function which was added to the COCONUTWLatentThinking.py file:

from COCONUTWLatentThinking import evaluate_with_livebench

# Evaluate your model with default settings
results, metrics = evaluate_with_livebench(your_model)

# Or specify custom paths
results, metrics = evaluate_with_livebench(
    model=your_model,
    questions_file="livebench/brain_chimera_livebench_fixed.jsonl",
    output_path="livebench/custom_results.json"
)
Option 2: Using the LiveBenchEvaluator Class
For more control over the evaluation process, you can use the LiveBenchEvaluator class directly:

from COCONUTWLatentThinking import LiveBenchEvaluator

# Initialize the evaluator
evaluator = LiveBenchEvaluator(
    model=your_model,
    questions_file="livebench/brain_chimera_livebench_fixed.jsonl",
    output_path="livebench/brain_chimera_results.json"
)

# Run the evaluation
results, metrics = evaluator.run_full_evaluation()
Understanding Results
The evaluation results contain:

Overall Score: The average score across all categories
Category Scores: Performance in each of the six categories
Task Scores: Detailed performance on specific tasks within each category
Processing Time: Average time taken to process questions
A higher score indicates better performance in that category or task.

Example Output
===== LIVEBENCH EVALUATION RESULTS =====
Overall score: 0.4285 (120 questions)
Average processing time: 3.75 seconds per question

--- BY CATEGORY ---
coding: 0.5250 (20 questions, 4.21s avg)
data_analysis: 0.3750 (20 questions, 5.35s avg)
instruction_following: 0.6500 (20 questions, 2.87s avg)
language: 0.3250 (20 questions, 3.12s avg)
math: 0.4000 (20 questions, 4.53s avg)
reasoning: 0.2960 (20 questions, 2.45s avg)

--- BY TASK ---
AMPS_Hard: 0.4545 (11 questions, 5.12s avg)
LCB_generation: 0.6000 (10 questions, 3.78s avg)
coding_completion: 0.4500 (10 questions, 4.63s avg)
...
Interpreting Performance
High Scores (>0.7): Excellent performance in that category
Medium Scores (0.4-0.7): Acceptable performance, but room for improvement
Low Scores (<0.4): Areas that need significant improvement
Look for patterns in the results to guide your model training:

If math scores are low but language scores are high, focus more on math training
If certain tasks within a category perform worse than others, target those specific tasks
Compare performance across categories to identify relative strengths and weaknesses
Customizing Evaluation
You can evaluate specific subsets of LiveBench by creating custom question files:

# Evaluate only math questions
evaluator = LiveBenchEvaluator(
    model=your_model,
    questions_file="livebench/custom_math_only.jsonl",
    output_path="livebench/math_results.json"
)
results, metrics = evaluator.run_full_evaluation()
Troubleshooting
If you encounter issues:

Model Interface Error: Ensure your model exposes a compatible interface (forward, generate, or generate_text method)
JSON Format Issues: Run the fix_json_format.py script on your questions file
Missing Dependencies: Verify all required packages are installed
Memory Issues: Reduce batch size or evaluate fewer questions at once
Conclusion
LiveBench provides an objective, contamination-free way to evaluate your BrainChimera model's performance across multiple capabilities. Use the results to guide your training and improvement efforts, focusing on categories and tasks where the model performs poorly.

By regularly evaluating with LiveBench as you develop your model, you can track improvements and ensure that your training methods are working as expected. EOF