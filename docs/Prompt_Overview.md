# AutoInterp Agent Prompts

This document explains how prompts are used throughout the AutoInterp Agent system. All prompts are now centralized in the `prompts/` directory for transparency and configurability.

## Prompt Organization

Prompts are organized by component:

1. **analysis_generator.yaml** - Prompts for generating analysis code
2. **analysis_planner.yaml** - Prompts for generating plans for analysis
3. **evaluator.yaml** - Prompts for evaluating analysis results
4. **question_manager.yaml** - Prompts for generating and prioritizing questions
5. **reporter.yaml** - Prompts for generating the final research report
6. **visualization_generator** - Prompts for generating visualization code
7. **visualization_planner** - Prompts for generating plans for a visualization

The main `prompts.yaml` file imports all individual prompt files.

Users can edit prompts directly in the .yaml files.

## Available Prompts

### Analysis Generator Prompts
- `system_message`: System message for code generation
- `base_template`: Base template for analysis specification
- `insights_template`: Template for including previous insights
- `retry_template`: Template for retry attempts
- `retry_instructions`: Instructions for avoiding errors in retries

### Evaluator Prompts
- `system_message`: System message for analysis evaluation
- `prompt_template`: Template for evaluating analysis results
- `follow_up_template`: Template for follow-up analysis recommendations

### Analysis Planner Prompts
- `system`: System message that defines the planner's role in AI interpretability research
- `user`: Template for generating analysis plans based on question and evaluation history
  - Variables:
    - `{{question}}`: The prioritized question text
    - `{{evaluations}}`: Compilation of previous evaluation results

### Question Manager Prompts
- `generator.system_message`: System message for question generation
- `generator.prompt_template`: Template for generating questions
- `prioritizer.system_message`: System message for question prioritization
- `prioritizer.prompt_template`: Template for prioritizing questions

### Code Generation Prompts
- `generation.system_message`: System message for code generation
- `generation.prompt_template`: Template for code generation
- `generation.context_template`: Template for including context
- `analysis.system_message`: System message for code analysis
- `analysis.prompt_template`: Template for code analysis
- `improvement.system_message`: System message for code improvement
- `improvement.prompt_template`: Template for code improvement

## Template Variables

Each prompt template uses specific variables that are filled in at runtime. The variables are organized by component:

### Analysis Generator Variables
- `{question_text}` - The complete question text
- `{model_name}` - The name of the model being analyzed
- `{model_path}` - The path to the model
- `{previous_insights}` - Previous insights from the research
- `{failed_script}` - Content of the script that failed (used in retry template)
- `{error_message}` - Error message from failed execution (used in retry template)
- `{error_traceback}` - Error traceback from failed execution (used in retry template)
- `{attempt}` - The current retry attempt number

### Analysis Planner Variables
- `{{question}}` - The prioritized question text
- `{{evaluations}}` - Compilation of previous evaluation results

### Evaluator Variables
- `{question_statement}` - The current question statement
- `{question_rationale}` - The rationale for the current question
- `{question_confidence}` - The current confidence level in the question
- `{script_content}` - Content of the analysis script
- `{stdout}` - Standard output from the analysis
- `{stderr_section}` - Error output section (if any)
- `{analysis_json}` - Analysis results in JSON format

### Question Manager Variables
- `{task_description}` - Description of the current task
- `{count}` - Number of questions to generate
- `{question_text}` - The complete question text for prioritization

Note that the Analysis Planner uses double curly braces template variables (e.g., `{{question}}`) that are replaced directly in the code, rather than using Python's `.format()` method.

For other prompts, variables are processed with Python's string `.format()` method.