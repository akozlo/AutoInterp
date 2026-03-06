"""
Report Generator module for the AutoInterp Agent Framework.
Creates comprehensive and reproducible interpretability reports.
"""

import os
import sys
import json
import re
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import markdown
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    weasyprint = None
    WEASYPRINT_AVAILABLE = False

from ..core.utils import ensure_directory, get_timestamp, load_yaml, PathResolver, get_comprehensive_log_path
from ..core.llm_interface import LLMInterface

# Core imports for mechanistic interpretability analyses (matches AnalysisExecutor)
NOTEBOOK_STANDARD_IMPORTS = """# --- Environment Setup: All dependencies for mechanistic interpretability ---
import os
import sys
import json
import random
import logging
import warnings
warnings.filterwarnings("ignore", "Can't initialize NVML")

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import sklearn
from sklearn import metrics, decomposition
from tqdm.auto import tqdm
import einops

# Mechanistic interpretability packages
import transformer_lens
from transformer_lens import HookedTransformer
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

# Parameters dict (some analysis scripts expect this from the executor)
parameters = {}
"""


class ReportGenerator:
    """
    Creates comprehensive reports from analysis results and visualizations.
    
    Responsibilities:
    - Generate structured reports from analysis results
    - Integrate visualizations into reports
    - Create reproducible documentation of findings
    - Support multiple output formats (markdown, Jupyter notebooks, HTML)
    """
    
    def __init__(self, config: Dict[str, Any], llm_interface: Optional[LLMInterface] = None):
        """
        Initialize the Report Generator.
        
        Args:
            config: Configuration dictionary
            llm_interface: Optional LLM interface for generating report content
        """
        self.config = config
        self.llm_interface = llm_interface
        
        # Use the PathResolver singleton instead of storing project_id directly
        self.path_resolver = PathResolver(config)
        
        # Log initialization with path resolver
        self.logger = logging.getLogger("autointerp.report_generator")
        self.logger.info(f"ReportGenerator initialized using path resolver with project_id: {self.path_resolver.project_id}")
        
        # Ensure reports directory exists and set output_dir
        self.output_dir = self.path_resolver.ensure_path("reports")
        
        # Log the output directory
        self.logger.debug(f"Reports directory: {self.output_dir}")
        
        # Configure report settings
        report_config = config.get("reporting", {})
        self.default_format = report_config.get("default_format", "markdown")
        
        # Load reporter prompts if LLM interface is available
        self.reporter_prompts = {}
        if self.llm_interface:
            try:
                # Load reporter prompts from the prompts configuration
                prompts_config = config.get("prompts", {})
                if prompts_config and "reporter" in prompts_config:
                    self.reporter_prompts = prompts_config["reporter"]
                    self.logger.info(f"Loaded {len(self.reporter_prompts)} reporter prompts")
                    self.logger.debug(f"Available reporter prompts: {list(self.reporter_prompts.keys())}")
                elif prompts_config:
                    # Fallback: try to load from the root level (for backward compatibility)
                    self.reporter_prompts = prompts_config
                    self.logger.info(f"Loaded {len(self.reporter_prompts)} prompts from root level")
                    self.logger.debug(f"Available prompts: {list(self.reporter_prompts.keys())}")
                else:
                    self.logger.warning("No prompts found in configuration")
            except Exception as e:
                self.logger.error(f"Failed to load reporter prompts: {e}")
        else:
            self.logger.warning("No LLM interface provided - will use basic report generation")
    
    def _get_visualization_files(self) -> List[str]:
        """
        Get all visualization files in the visualizations directory.
        
        Returns:
            List of visualization filenames
        """
        try:
            visualizations_dir = self.path_resolver.get_path("visualizations")
            if not visualizations_dir.exists():
                self.logger.warning(f"Visualizations directory not found at {visualizations_dir}")
                return []
            
            # Look for common image extensions
            image_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
            visualization_files = []
            
            for ext in image_extensions:
                visualization_files.extend(visualizations_dir.glob(f"*{ext}"))
            
            # Return just the filenames, sorted for consistency
            filenames = [f.name for f in visualization_files]
            filenames.sort()
            
            self.logger.info(f"Found {len(filenames)} visualization files: {filenames}")
            return filenames
            
        except Exception as e:
            self.logger.error(f"Failed to get visualization files: {e}")
            return []
    
    def _load_comprehensive_log(self) -> str:
        """
        Load the comprehensive log content.
        
        Returns:
            String content of the comprehensive log file, or empty string if not found
        """
        try:
            project_dir = self.path_resolver.get_project_dir()
            comprehensive_log_path = get_comprehensive_log_path(project_dir)
            
            if comprehensive_log_path.exists():
                with open(comprehensive_log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.logger.info(f"Loaded comprehensive log from {comprehensive_log_path}")
                return content
            else:
                self.logger.warning(f"Comprehensive log not found at {comprehensive_log_path}")
                return ""
        except Exception as e:
            self.logger.error(f"Failed to load comprehensive log: {e}")
            return ""
    
    def _generate_fallback_title(self, question: Union[str, Dict[str, Any]], 
                                evaluation_results: Dict[str, Any], timestamp: str) -> str:
        """
        Generate a fallback title using the original logic.
        
        Args:
            question: Question dictionary or string
            evaluation_results: Evaluation results dictionary
            timestamp: Current timestamp string
            
        Returns:
            Generated title string
        """
        # First check if there's a project title in config
        project_title = self.config.get("title")
        if project_title:
            conclusion = evaluation_results.get("conclusion", "INCONCLUSIVE")
            return f"Interpretability Report: {project_title} - Question {conclusion}"
        # Handle None question
        elif question is None:
            return f"Interpretability Report {timestamp}"
        # Handle string question (plain text)
        elif isinstance(question, str):
            # Look for "QUESTION:" line in raw text
            hyp_match = re.search(r'QUESTION:\s*(.*?)(?:\n|$)', question, re.IGNORECASE)
            if hyp_match:
                hyp_statement = hyp_match.group(1).strip()
            else:
                # Just use first line as statement for title
                lines = question.strip().split('\n')
                hyp_statement = lines[0] if lines else "Unnamed Question"
            
            if len(hyp_statement) > 50:
                hyp_statement = hyp_statement[:47] + "..."
            
            conclusion = evaluation_results.get("conclusion", "INCONCLUSIVE")
            return f"Interpretability Report: {hyp_statement} - Question {conclusion}"
        # Otherwise try to extract from raw text field
        elif "raw_text" in question:
            # Look for "QUESTION:" line in raw text
            raw_text = question.get("raw_text", "")
            hyp_match = re.search(r'QUESTION:\s*(.*?)(?:\n|$)', raw_text, re.IGNORECASE)
            if hyp_match:
                hyp_statement = hyp_match.group(1).strip()
            else:
                # Just use first line as statement for title
                lines = raw_text.strip().split('\n')
                hyp_statement = lines[0] if lines else "Unnamed Question"
            
            if len(hyp_statement) > 50:
                hyp_statement = hyp_statement[:47] + "..."
            
            conclusion = evaluation_results.get("conclusion", "INCONCLUSIVE")
            return f"Interpretability Report: {hyp_statement} - Question {conclusion}"
        else:
            return f"Interpretability Report {timestamp}"
    
    async def generate_report(self,
                       question: Union[str, Dict[str, Any]],  # Can be plain text or dictionary
                       analysis_results: Dict[str, Any],
                       evaluation_results: Dict[str, Any],
                       visualizations: Dict[str, str],
                       title: Optional[str] = None,
                       output_format: Optional[str] = None) -> str:
        """
        Generate a comprehensive report from analysis results.
        
        Args:
            question: Question dictionary
            analysis_results: Analysis results dictionary
            evaluation_results: Evaluation results dictionary
            visualizations: Dictionary mapping visualization types to file paths
            title: Optional report title
            output_format: Output format (markdown, jupyter, html)
            
        Returns:
            Path to the generated report
        """
        # Set defaults
        output_format = output_format or self.default_format
        timestamp = get_timestamp().replace(":", "-").replace(" ", "_")
        
        # Generate a title if none is provided
        if not title:
            # Try to generate title using LLM if available
            self.logger.info(f"Attempting to generate title. LLM available: {bool(self.llm_interface)}")
            self.logger.info(f"Available prompts: {list(self.reporter_prompts.keys())}")
            if self.llm_interface and "title_generator" in self.reporter_prompts:
                try:
                    # Load comprehensive log for title generation
                    comprehensive_log = self._load_comprehensive_log()
                    self.logger.info(f"Loaded comprehensive log with {len(comprehensive_log)} characters")
                    
                    generated_title = await self._generate_section_content(
                        "title_generator",
                        comprehensive_log=comprehensive_log
                    )
                    
                    self.logger.info(f"Raw generated title: '{generated_title}'")
                    
                    # Use the generated title if it's valid
                    if generated_title and generated_title.strip() and not generated_title.startswith("<!--"):
                        title = generated_title.strip()
                        self.logger.info(f"Using LLM-generated title: {title}")
                    else:
                        self.logger.warning(f"Invalid generated title, falling back. Generated: '{generated_title}'")
                        # Fallback to default title generation
                        title = self._generate_fallback_title(question, evaluation_results, timestamp)
                except Exception as e:
                    self.logger.error(f"Error generating title with LLM: {e}")
                    # Fallback to default title generation
                    title = self._generate_fallback_title(question, evaluation_results, timestamp)
            else:
                # Fallback to default title generation
                title = self._generate_fallback_title(question, evaluation_results, timestamp)
        
        # Get the latest output directory path (may have changed if project was renamed)
        output_dir = self.path_resolver.ensure_path("reports")
        
        # Generate a clean filename based on the title or study name
        filename = self._get_filename_from_title(title)
        
        # Generate the appropriate report format
        # Note: Jupyter notebook is generated separately via generate_jupyter_notebook in main.py
        if output_format == "html":
            report_path = await self._generate_html_report(
                question=question,
                analysis_results=analysis_results,
                evaluation_results=evaluation_results,
                visualizations=visualizations,
                title=title,
                output_path=output_dir / f"{filename}.html"
            )
        elif output_format == "pdf":
            report_path = await self._generate_pdf_report(
                question=question,
                analysis_results=analysis_results,
                evaluation_results=evaluation_results,
                visualizations=visualizations,
                title=title,
                output_path=output_dir / f"{filename}.pdf"
            )
        else:  # Default to markdown
            report_path = await self._generate_markdown_report(
                question=question,
                analysis_results=analysis_results,
                evaluation_results=evaluation_results,
                visualizations=visualizations,
                title=title,
                output_path=output_dir / f"{filename}.md"
            )
        
        self.logger.info(f"Generated report at {report_path}")
        return str(report_path)
    
    async def _generate_section_content(self, section_name: str, **kwargs) -> str:
        """
        Generate content for a specific report section using LLM prompts.
        
        Args:
            section_name: Name of the section prompt to use
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Generated section content
        """
        if not self.llm_interface or section_name not in self.reporter_prompts:
            return f"<!-- Section {section_name} not available -->"
        
        try:
            prompt_config = self.reporter_prompts[section_name]
            system_prompt = prompt_config.get("system_prompt", "")
            user_prompt_template = prompt_config.get("user_prompt", "")
            
            # Substitute variables in the user prompt
            user_prompt = user_prompt_template.format(**kwargs)
            
            # Generate content using LLM
            # Use the section name as agent name if it has its own config, otherwise default to reporter
            agent_name = section_name if section_name in ["title_generator"] else "reporter"
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=system_prompt,
                agent_name=agent_name
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating {section_name} section: {e}")
            return f"<!-- Error generating {section_name} section: {e} -->"
    
    def _extract_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Extract a structured summary of analysis outputs for report sections."""
        analyses = analysis_results.get("analyses", [])
        if not analyses:
            return "No analyses conducted."
        
        summaries = []
        for i, analysis in enumerate(analyses, 1):
            stdout = analysis.get("results", {}).get("stdout", "") or analysis.get("stdout", "")
            if stdout:
                # Extract key findings more intelligently
                lines = stdout.split('\n')
                
                # Look for specific result patterns
                key_findings = []
                for line in lines:
                    line = line.strip()
                    if any(pattern in line.lower() for pattern in ['peak accuracy:', 'critical layers:', 'question evaluation:', 'conclusion:']):
                        key_findings.append(line)
                    elif 'accuracy' in line.lower() and any(layer_pattern in line for layer_pattern in ['layer', 'Layer']):
                        # Extract layer accuracy info
                        if len(key_findings) < 2:  # Limit to avoid overwhelming
                            key_findings.append(line)
                
                if key_findings:
                    summaries.append(f"**Analysis {i}:** " + " | ".join(key_findings[:3]))
                else:
                    # Fallback to extracting just conclusion-type lines
                    conclusion_lines = [line for line in lines if any(keyword in line.lower() for keyword in 
                                      ['result:', 'conclusion', 'question', 'supported', 'refuted'])]
                    if conclusion_lines:
                        summaries.append(f"**Analysis {i}:** {conclusion_lines[0]}")
                    else:
                        summaries.append(f"**Analysis {i}:** Analysis completed successfully")
            else:
                summaries.append(f"**Analysis {i}:** No output captured")
        
        return "\n\n".join(summaries)
    
    def _extract_key_metrics(self, analysis_results: Dict[str, Any]) -> str:
        """Extract key quantitative metrics from analysis outputs."""
        analyses = analysis_results.get("analyses", [])
        if not analyses:
            return "No metrics available."
        
        all_metrics = []
        for i, analysis in enumerate(analyses, 1):
            stdout = analysis.get("results", {}).get("stdout", "") or analysis.get("stdout", "")
            if stdout:
                import re
                metrics_for_analysis = []
                
                # Extract peak accuracy values
                peak_matches = re.findall(r'peak accuracy:\s*([0-9.]+)', stdout, re.IGNORECASE)
                if peak_matches:
                    metrics_for_analysis.append(f"Peak accuracy: {peak_matches[0]}")
                
                # Extract critical layers information
                critical_layer_matches = re.findall(r'critical layers[^:]*:\s*\[([^\]]*)\]', stdout, re.IGNORECASE)
                if critical_layer_matches:
                    metrics_for_analysis.append(f"Critical layers: [{critical_layer_matches[0]}]")
                
                # Extract final confidence if mentioned
                confidence_matches = re.findall(r'confidence[^:]*:\s*([0-9.]+)', stdout, re.IGNORECASE)
                if confidence_matches:
                    metrics_for_analysis.append(f"Final confidence: {confidence_matches[-1]}")
                
                # Extract layer accuracy ranges
                layer_acc_matches = re.findall(r'Layer\s+(\d+):\s+Accuracy\s*=\s*([0-9.]+)', stdout)
                if layer_acc_matches:
                    accuracies = [float(acc) for _, acc in layer_acc_matches]
                    if accuracies:
                        metrics_for_analysis.append(f"Accuracy range: {min(accuracies):.3f}-{max(accuracies):.3f}")
                
                if metrics_for_analysis:
                    all_metrics.append(f"**Analysis {i}:** " + " | ".join(metrics_for_analysis))
                else:
                    all_metrics.append(f"**Analysis {i}:** No quantitative metrics extracted")
        
        return "\n".join(all_metrics) if all_metrics else "No quantitative metrics extracted."

    async def _generate_markdown_report(self,
                                question: Union[str, Dict[str, Any]],
                                analysis_results: Dict[str, Any],
                                evaluation_results: Dict[str, Any],
                                visualizations: Dict[str, str],
                                title: str,
                                output_path: Path) -> str:
        """
        Generate a markdown report.
        
        Args:
            question: Question dictionary
            analysis_results: Analysis results dictionary
            evaluation_results: Evaluation results dictionary
            visualizations: Dictionary mapping visualization types to file paths
            title: Report title
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Prepare context variables for LLM generation
        question_text = ""
        if question is None:
            question_text = "No question available"
        elif isinstance(question, str):
            question_text = question
        elif "raw_text" in question:
            question_text = question.get("raw_text", "No question text provided")
        else:
            question_text = question.get("statement", "No question statement provided")
        
        # Extract model name and task description
        model_name = self.config.get("model", {}).get("name", "Unknown Model")
        task_description = self.config.get("task", {}).get("description", "Interpretability research")
        
        # Extract analysis plan
        analysis_plan_text = "No analysis plan provided."
        latest_analysis = analysis_results.get("latest_analysis", {})
        if "analysis_plan" in latest_analysis:
            analysis_plan_text = latest_analysis["analysis_plan"]
        else:
            try:
                plans_dir = self.path_resolver.get_path("analysis_plans")
                if plans_dir.exists():
                    plan_files = sorted(plans_dir.glob("analysis_plan_*.txt"), key=lambda x: x.name, reverse=True)
                    if plan_files:
                        with open(plan_files[0], 'r') as f:
                            analysis_plan_text = f.read()
            except Exception as e:
                self.logger.warning(f"Could not read analysis plan file: {e}")
        
        # Extract evaluation findings
        evaluation_findings = ""
        all_evaluations = evaluation_results.get("evaluations", [])
        if all_evaluations:
            latest_eval = all_evaluations[-1]
            evaluation_findings = (latest_eval.get("evaluation_text") or 
                                 latest_eval.get("explanation") or 
                                 latest_eval.get("raw_evaluation", "No evaluation findings available"))
        
        # Build the report content
        report_content = []
        
        # Header
        report_content.append(f"# {title}")
        report_content.append("")
        
        # Abstract - always include with fallback
        report_content.append("## Abstract")
        if self.llm_interface and "abstract" in self.reporter_prompts:
            try:
                # Load comprehensive log for abstract generation
                comprehensive_log = self._load_comprehensive_log()

                final_confidence = evaluation_results.get("final_confidence", 0.0)
                initial_confidence = 0.0  # Default since we're using raw text
                conclusion = evaluation_results.get("conclusion", "inconclusive").upper()

                abstract = await self._generate_section_content(
                    "abstract",
                    comprehensive_log=comprehensive_log
                )
                report_content.append(abstract)
            except Exception as e:
                self.logger.error(f"Error generating abstract: {e}")
                # Fallback executive summary
                final_confidence = evaluation_results.get("final_confidence", 0.0)
                conclusion = evaluation_results.get("conclusion", "inconclusive").upper()
                analysis_count = analysis_results.get("analysis_count", len(analysis_results.get("analyses", [])))
                report_content.append(f"This study investigated the question: {question_text[:150]}{'...' if len(question_text) > 150 else ''}")
                report_content.append(f"After {analysis_count} analyses, the results were {conclusion} with {final_confidence:.2f} confidence.")
        else:
            # Basic fallback executive summary
            final_confidence = evaluation_results.get("final_confidence", 0.0)
            conclusion = evaluation_results.get("conclusion", "inconclusive").upper()
            analysis_count = analysis_results.get("analysis_count", len(analysis_results.get("analyses", [])))
            report_content.append(f"This study investigated the question: {question_text[:150]}{'...' if len(question_text) > 150 else ''}")
            report_content.append(f"After {analysis_count} analyses, the results were {conclusion} with {final_confidence:.2f} confidence.")
        report_content.append("")
        
        # Introduction Section - always include with fallback
        report_content.append("## Introduction")
        if self.llm_interface and "introduction" in self.reporter_prompts:
            try:
                # Load comprehensive log for introduction generation
                comprehensive_log = self._load_comprehensive_log()
                
                introduction = await self._generate_section_content(
                    "introduction",
                    comprehensive_log=comprehensive_log
                )
                report_content.append(introduction)
            except Exception as e:
                self.logger.error(f"Error generating introduction: {e}")
                # Fallback introduction
                report_content.append(f"This study examines interpretability aspects of the {model_name} model, specifically investigating:")
                report_content.append("")
                report_content.append(f"**Question:** {question_text}")
                report_content.append("")
                report_content.append(f"**Research Context:** {task_description}")
        else:
            # Basic fallback introduction
            report_content.append(f"This study examines interpretability aspects of the {model_name} model, specifically investigating:")
            report_content.append("")
            report_content.append(f"**Question:** {question_text}")
            report_content.append("")
            report_content.append(f"**Research Context:** {task_description}")
        report_content.append("")
        
        # Methodology Section - always include with fallback
        report_content.append("## Methodology")
        if self.llm_interface and "methodology_summary" in self.reporter_prompts:
            try:
                # Load comprehensive log content
                comprehensive_log = self._load_comprehensive_log()
                
                methodology = await self._generate_section_content(
                    "methodology_summary",
                    comprehensive_log=comprehensive_log
                )
                report_content.append(methodology)
            except Exception as e:
                self.logger.error(f"Error generating methodology: {e}")
                # Fallback methodology
                report_content.append("### Analysis Approach")
                report_content.append(analysis_plan_text if analysis_plan_text != "No analysis plan provided." else "The study employed systematic computational analysis of model internals.")
                if self.config.get("analysis"):
                    report_content.append("")
                    report_content.append("### Analysis Parameters")
                    for key, value in self.config.get("analysis", {}).items():
                        report_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        else:
            # Basic fallback methodology
            report_content.append("### Analysis Approach")
            report_content.append(analysis_plan_text if analysis_plan_text != "No analysis plan provided." else "The study employed systematic computational analysis of model internals.")
            if self.config.get("analysis"):
                report_content.append("")
                report_content.append("### Analysis Parameters")
                for key, value in self.config.get("analysis", {}).items():
                    report_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        report_content.append("")
        
        # Results Section - always include with fallback
        report_content.append("## Results")
        if self.llm_interface and "results_synthesis" in self.reporter_prompts:
            try:
                # Load comprehensive log content
                comprehensive_log = self._load_comprehensive_log()
                
                # Get visualization files list
                visualization_files = self._get_visualization_files()
                
                results_section = await self._generate_section_content(
                    "results_synthesis",
                    comprehensive_log=comprehensive_log,
                    visualization_files=visualization_files
                )
                report_content.append(results_section)
            except Exception as e:
                self.logger.error(f"Error generating results section: {e}")
                # Fallback to basic results section
                report_content.append("### Analysis Outputs")
                report_content.append(self._extract_analysis_summary(analysis_results))
                report_content.append("")
                report_content.append("### Key Metrics")
                report_content.append(self._extract_key_metrics(analysis_results))
        else:
            # Basic fallback results section
            report_content.append("### Analysis Outputs")
            report_content.append(self._extract_analysis_summary(analysis_results))
            report_content.append("")
            report_content.append("### Key Metrics")
            report_content.append(self._extract_key_metrics(analysis_results))
        report_content.append("")
        
        # Discussion and Conclusion Section - always include with fallback
        report_content.append("## Conclusion")
        final_confidence = evaluation_results.get("final_confidence", 0.0)
        initial_confidence = 0.0  # Default since we're using raw text
        confidence_change = final_confidence - initial_confidence
        conclusion_text = evaluation_results.get("conclusion", "inconclusive").upper()
        analysis_count = analysis_results.get("analysis_count", len(analysis_results.get("analyses", [])))
        
        if self.llm_interface and "discussion_and_conclusion" in self.reporter_prompts:
            try:
                # Extract key insights
                key_insights = []
                all_evaluations = evaluation_results.get("evaluations", [])
                for eval_result in all_evaluations:
                    if "key_insights" in eval_result and eval_result["key_insights"]:
                        key_insights.extend(eval_result["key_insights"])
                
                # Load comprehensive log content
                comprehensive_log = self._load_comprehensive_log()
                
                discussion_conclusion = await self._generate_section_content(
                    "discussion_and_conclusion",
                    comprehensive_log=comprehensive_log
                )
                
                report_content.append(discussion_conclusion)
            except Exception as e:
                self.logger.error(f"Error generating discussion and conclusion: {e}")
                # Fallback discussion and conclusion
                report_content.append("## Discussion")
                report_content.append("### Interpretation of Results")
                report_content.append(f"The analysis of {analysis_count} separate investigations yielded {conclusion_text.lower()} evidence regarding the question.")
                report_content.append("")
                if evaluation_findings:
                    report_content.append("### Evaluation Findings")
                    # Split evaluation findings into paragraphs for better readability
                    findings_lines = evaluation_findings.split('\n')
                    for line in findings_lines:
                        if line.strip():
                            report_content.append(line.strip())
                    report_content.append("")
                
                report_content.append("### Confidence Assessment")
                report_content.append(f"The investigation began with a baseline confidence of {initial_confidence:.2f} and concluded with a final confidence of {final_confidence:.2f}, representing a change of {confidence_change:+.2f}.")
                
                if abs(confidence_change) > 0.1:
                    if confidence_change > 0:
                        report_content.append("This positive change suggests the evidence has helped to answer the question.")
                    else:
                        report_content.append("This negative change suggests the evidence has pushed us farther from answering the question.")
                else:
                    report_content.append("The minimal confidence change suggests ambiguous or inconclusive evidence.")
                report_content.append("")
                
                report_content.append("## Conclusion")
                if "confidence_statement" in evaluation_results:
                    report_content.append(evaluation_results["confidence_statement"])
                else:
                    if conclusion_text == "SUPPORT":
                        report_content.append(f"Based on {analysis_count} comprehensive analysis(es), the evidence **SUPPORTS** the question with a final confidence of {final_confidence:.2f}.")
                        report_content.append("The findings provide measurable evidence consistent with the predicted model behavior.")
                    elif conclusion_text == "REFUTATION": 
                        report_content.append(f"Based on {analysis_count} comprehensive analysis(es), the evidence **REFUTES** the question with a final confidence of {final_confidence:.2f}.")
                        report_content.append("The findings contradict the predicted model behavior, suggesting alternative mechanisms.")
                    else:
                        report_content.append(f"Based on {analysis_count} comprehensive analysis(es), the evidence is **INCONCLUSIVE** regarding the question with a final confidence of {final_confidence:.2f}.")
                        report_content.append("The findings do not help us answer the research question, indicating the need for additional investigation or refined methodology.")
        else:
            # Basic fallback discussion and conclusion when no LLM interface available
            report_content.append("## Discussion")
            report_content.append("### Interpretation of Results")
            report_content.append(f"The analysis of {analysis_count} separate investigations yielded {conclusion_text.lower()} evidence regarding the question.")
            report_content.append("")
            
            if evaluation_findings:
                report_content.append("### Evaluation Findings")
                # Split evaluation findings into paragraphs for better readability
                findings_lines = evaluation_findings.split('\n')
                for line in findings_lines:
                    if line.strip():
                        report_content.append(line.strip())
                report_content.append("")
            
            report_content.append("### Confidence Assessment")
            report_content.append(f"The investigation began with a baseline confidence of {initial_confidence:.2f} and concluded with a final confidence of {final_confidence:.2f}, representing a change of {confidence_change:+.2f}.")
            
            if abs(confidence_change) > 0.1:
                if confidence_change > 0:
                    report_content.append("This positive change suggests the evidence improved our confidence in our answer to the research question.")
                else:
                    report_content.append("This negative change suggests the evidence challenged or contradicted our previous answer to the research question.")
            else:
                report_content.append("The minimal confidence change suggests ambiguous or inconclusive evidence.")
            report_content.append("")
            
            report_content.append("## Conclusion")
            if "confidence_statement" in evaluation_results:
                report_content.append(evaluation_results["confidence_statement"])
            else:
                if conclusion_text == "SUPPORT":
                    report_content.append(f"Based on {analysis_count} comprehensive analysis(es), we have settled on an answer to the research question with a final confidence of {final_confidence:.2f}.")
                    report_content.append("The findings provide measurable evidence consistent with the predicted model behavior.")
                else:
                    report_content.append(f"Based on {analysis_count} comprehensive analysis(es), the evidence is **INCONCLUSIVE** regarding the question with a final confidence of {final_confidence:.2f}.")
                    report_content.append("The findings do not conclusively answer the research question, indicating the need for additional investigation or refined methodology.")
        
        report_content.append("")
        
        # Footer
        report_content.append("---")
        report_content.append("*This report was automatically generated by the AutoInterp Agent Framework.*")
        
        # Save the report
        with open(output_path, "w") as f:
            f.write("\n".join(report_content))
        
        return str(output_path)
    
    async def _generate_html_report(self,
                            question: Union[str, Dict[str, Any]],
                            analysis_results: Dict[str, Any],
                            evaluation_results: Dict[str, Any],
                            visualizations: Dict[str, str],
                            title: str,
                            output_path: Path) -> str:
        """
        Generate an HTML report.
        
        Args:
            question: Question dictionary
            analysis_results: Analysis results dictionary
            evaluation_results: Evaluation results dictionary
            visualizations: Dictionary mapping visualization types to file paths
            title: Report title
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # First generate markdown content
        # Use output_path's parent directory to ensure temp file is in the same directory
        md_path = output_path.parent / f"{self._sanitize_filename(title)}_temp.md"
        await self._generate_markdown_report(
            question=question,
            analysis_results=analysis_results,
            evaluation_results=evaluation_results,
            visualizations=visualizations,
            title=title,
            output_path=md_path
        )
        
        # Read the markdown content
        with open(md_path, "r") as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add HTML styling
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        code, pre {{
            background-color: #f5f7f9;
            border-radius: 3px;
            padding: 2px 5px;
            font-family: monospace;
        }}
        pre {{
            padding: 10px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        th {{
            background-color: #f2f2f2;
            text-align: left;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #777;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
        
        # Save the HTML file
        with open(output_path, "w") as f:
            f.write(styled_html)
        
        # Clean up the temporary markdown file
        os.remove(md_path)
        
        return str(output_path)
    
    async def _generate_pdf_report(self,
                            question: Union[str, Dict[str, Any]],
                            analysis_results: Dict[str, Any],
                            evaluation_results: Dict[str, Any],
                            visualizations: Dict[str, str],
                            title: str,
                            output_path: Path) -> str:
        """
        Generate a PDF report.
        
        Args:
            question: Question dictionary
            analysis_results: Analysis results dictionary
            evaluation_results: Evaluation results dictionary
            visualizations: Dictionary mapping visualization types to file paths
            title: Report title
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # First generate HTML content
        # Use output_path's parent directory to ensure temp file is in the same directory
        html_path = output_path.parent / f"{self._sanitize_filename(title)}_temp.html"
        await self._generate_html_report(
            question=question,
            analysis_results=analysis_results,
            evaluation_results=evaluation_results,
            visualizations=visualizations,
            title=title,
            output_path=html_path
        )
        
        if not WEASYPRINT_AVAILABLE:
            self.logger.error("weasyprint not available for PDF generation")
            # Clean up the temporary HTML file if it exists
            if html_path.exists():
                os.remove(html_path)
            raise Exception("PDF generation requires weasyprint package. Please install it with: pip install weasyprint")

        try:
            # Convert HTML to PDF using weasyprint
            html_doc = weasyprint.HTML(filename=str(html_path))
            
            # Enhanced CSS for better PDF rendering
            css = weasyprint.CSS(string="""
                @page {
                    margin: 1in;
                    size: letter;
                }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    font-size: 11pt;
                }
                h1 {
                    color: #2c3e50;
                    font-size: 24pt;
                    margin-top: 0;
                    page-break-before: avoid;
                }
                h2 {
                    color: #2c3e50;
                    font-size: 18pt;
                    margin-top: 20pt;
                    margin-bottom: 10pt;
                    page-break-after: avoid;
                }
                h3 {
                    color: #2c3e50;
                    font-size: 14pt;
                    margin-top: 15pt;
                    margin-bottom: 8pt;
                    page-break-after: avoid;
                }
                h4 {
                    color: #2c3e50;
                    font-size: 12pt;
                    margin-top: 12pt;
                    margin-bottom: 6pt;
                }
                p {
                    margin-bottom: 8pt;
                    text-align: justify;
                }
                code, pre {
                    background-color: #f5f7f9;
                    border-radius: 3px;
                    padding: 2pt 4pt;
                    font-family: "Courier New", Courier, monospace;
                    font-size: 9pt;
                }
                pre {
                    padding: 8pt;
                    overflow: hidden;
                    margin: 8pt 0;
                    border: 1pt solid #ddd;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    margin: 8pt 0;
                    page-break-inside: avoid;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 12pt 0;
                    font-size: 10pt;
                    page-break-inside: avoid;
                }
                th, td {
                    border: 1pt solid #ddd;
                    padding: 6pt;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .page-break {
                    page-break-before: always;
                }
                .avoid-break {
                    page-break-inside: avoid;
                }
                blockquote {
                    border-left: 3pt solid #ddd;
                    margin: 8pt 0;
                    padding-left: 12pt;
                    font-style: italic;
                }
                ul, ol {
                    margin: 8pt 0;
                    padding-left: 20pt;
                }
                li {
                    margin-bottom: 4pt;
                }
            """)
            
            # Generate the PDF
            html_doc.write_pdf(str(output_path), stylesheets=[css])
            
            # Clean up the temporary HTML file
            os.remove(html_path)
            
            self.logger.info(f"Successfully generated PDF report at {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            # Clean up the temporary HTML file if it exists
            if html_path.exists():
                os.remove(html_path)
            
            # Fallback: rename HTML file to indicate it's the fallback
            fallback_path = output_path.parent / f"{output_path.stem}_fallback.html"
            if html_path.exists():
                os.rename(html_path, fallback_path)
                self.logger.warning(f"PDF generation failed, HTML fallback saved at {fallback_path}")
                return str(fallback_path)
            else:
                raise Exception(f"PDF generation failed and no fallback available: {e}")
    
    def generate_summary_report(self,
                              task_results: Dict[str, Any],
                              questions: List[Dict[str, Any]],
                              task_config: Dict[str, Any],
                              title: Optional[str] = None,
                              output_format: Optional[str] = None) -> str:
        """
        Generate a summary report for all questions in a task.
        
        Args:
            task_results: Task execution results
            questions: List of question dictionaries
            task_config: Task configuration dictionary
            title: Optional report title
            output_format: Output format (markdown, jupyter, html)
            
        Returns:
            Path to the generated report
        """
        # Set defaults
        output_format = output_format or self.default_format
        timestamp = get_timestamp().replace(":", "-").replace(" ", "_")
        title = title or f"Task Summary Report: Interpretability Study {timestamp}"
        
        # Get the latest output directory path (may have changed if project was renamed)
        output_dir = self.path_resolver.ensure_path("reports")
        
        # Generate the appropriate report format
        # Note: Jupyter format for summary reports is not supported; falls through to markdown
        if output_format == "html":
            report_path = self._generate_summary_html(
                task_results=task_results,
                questions=questions,
                task_config=task_config,
                title=title,
                output_path=output_dir / f"{self._sanitize_filename(title)}.html"
            )
        elif output_format == "pdf":
            report_path = self._generate_summary_pdf(
                task_results=task_results,
                questions=questions,
                task_config=task_config,
                title=title,
                output_path=output_dir / f"{self._sanitize_filename(title)}.pdf"
            )
        else:  # Default to markdown
            report_path = self._generate_summary_markdown(
                task_results=task_results,
                questions=questions,
                task_config=task_config,
                title=title,
                output_path=output_dir / f"{self._sanitize_filename(title)}.md"
            )
        
        self.logger.info(f"Generated summary report at {report_path}")
        return str(report_path)
    
    def _generate_summary_markdown(self,
                                 task_results: Dict[str, Any],
                                 questions: List[Dict[str, Any]],
                                 task_config: Dict[str, Any],
                                 title: str,
                                 output_path: Path) -> str:
        """
        Generate a markdown summary report.
        
        Args:
            task_results: Task execution results
            questions: List of question dictionaries
            task_config: Task configuration dictionary
            title: Report title
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # Build the report content
        report_content = []
        
        # Header
        report_content.append(f"# {title}")
        report_content.append("")
        
        # Task Information
        report_content.append("## Task Information")
        # Task name field removed - no longer needed
        
        if "description" in task_config:
            report_content.append(f"**Description**: {task_config['description']}")
        
        if "model" in task_config:
            model_info = task_config["model"]
            report_content.append(f"**Model**: {model_info.get('name', 'Unknown')}")
        
        report_content.append("")
        
        # Summary of Findings
        if "reporting" in task_results and "summary" in task_results["reporting"]:
            report_content.append("## Summary of Findings")
            report_content.append(task_results["reporting"]["summary"])
            report_content.append("")
        
        # Questions Overview
        report_content.append("## Questions Overview")
        
        if questions:
            # Create a table of questions and their confidence
            report_content.append("| # | Question | Initial Confidence | Final Confidence | Supported |")
            report_content.append("|---|-----------|-------------------|-----------------|-----------|")
            
            for i, hyp in enumerate(questions):
                # Extract info
                statement = hyp.get("statement", "No statement")
                initial_confidence = hyp.get("initial_confidence", hyp.get("original_confidence", 0.0))
                final_confidence = hyp.get("confidence", 0.0)
                supported = "" if hyp.get("supported", None) else "L" if hyp.get("supported", None) is False else "S"
                
                report_content.append(f"| {i+1} | {statement} | {initial_confidence:.2f} | {final_confidence:.2f} | {supported} |")
            
            report_content.append("")
        else:
            report_content.append("No questions were investigated.")
            report_content.append("")
        
        # Detailed Question Results
        report_content.append("## Detailed Question Results")
        
        if questions:
            for i, hyp in enumerate(questions):
                report_content.append(f"### Question {i+1}: {hyp.get('statement', 'No statement')}")
                
                if "rationale" in hyp:
                    report_content.append(f"**Rationale**: {hyp['rationale']}")
                
                report_content.append(f"**Initial Confidence**: {hyp.get('initial_confidence', hyp.get('original_confidence', 0.0))}")
                report_content.append(f"**Final Confidence**: {hyp.get('confidence', 0.0)}")
                report_content.append(f"**Supported**: {hyp.get('supported', 'Unknown')}")
                
                if "evidence" in hyp:
                    report_content.append(f"**Evidence**:")
                    
                    if isinstance(hyp["evidence"], list):
                        for evidence in hyp["evidence"]:
                            report_content.append(f"- {evidence}")
                    else:
                        report_content.append(hyp["evidence"])
                
                report_content.append("")
        else:
            report_content.append("No detailed question results available.")
            report_content.append("")
        
        # Open Questions
        if "reporting" in task_results and "open_questions" in task_results["reporting"]:
            report_content.append("## Open Questions")
            report_content.append(task_results["reporting"]["open_questions"])
            report_content.append("")
        
        # Footer
        report_content.append("---")
        report_content.append("*This report was automatically generated by the AutoInterp Agent Framework.*")
        
        # Save the report
        with open(output_path, "w") as f:
            f.write("\n".join(report_content))
        
        return str(output_path)
    
    def _generate_summary_html(self,
                             task_results: Dict[str, Any],
                             questions: List[Dict[str, Any]],
                             task_config: Dict[str, Any],
                             title: str,
                             output_path: Path) -> str:
        """
        Generate an HTML summary report.
        
        Args:
            task_results: Task execution results
            questions: List of question dictionaries
            task_config: Task configuration dictionary
            title: Report title
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # First generate markdown content
        # Use output_path's parent directory to ensure temp file is in the same directory
        md_path = output_path.parent / f"{self._sanitize_filename(title)}_temp.md"
        self._generate_summary_markdown(
            task_results=task_results,
            questions=questions,
            task_config=task_config,
            title=title,
            output_path=md_path
        )
        
        # Read the markdown content
        with open(md_path, "r") as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add HTML styling
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        code, pre {{
            background-color: #f5f7f9;
            border-radius: 3px;
            padding: 2px 5px;
            font-family: monospace;
        }}
        pre {{
            padding: 10px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        th {{
            background-color: #f2f2f2;
            text-align: left;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #777;
        }}
        .supported-true {{
            color: green;
            font-weight: bold;
        }}
        .supported-false {{
            color: red;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
        
        # Save the HTML file
        with open(output_path, "w") as f:
            f.write(styled_html)
        
        # Clean up the temporary markdown file
        os.remove(md_path)
        
        return str(output_path)
    
    def _generate_summary_pdf(self,
                            task_results: Dict[str, Any],
                            questions: List[Dict[str, Any]],
                            task_config: Dict[str, Any],
                            title: str,
                            output_path: Path) -> str:
        """
        Generate a PDF summary report.
        
        Args:
            task_results: Task execution results
            questions: List of question dictionaries
            task_config: Task configuration dictionary
            title: Report title
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        # First generate HTML content
        # Use output_path's parent directory to ensure temp file is in the same directory
        html_path = output_path.parent / f"{self._sanitize_filename(title)}_temp.html"
        self._generate_summary_html(
            task_results=task_results,
            questions=questions,
            task_config=task_config,
            title=title,
            output_path=html_path
        )
        
        if not WEASYPRINT_AVAILABLE:
            self.logger.error("weasyprint not available for PDF generation")
            # Clean up the temporary HTML file if it exists
            if html_path.exists():
                os.remove(html_path)
            raise Exception("PDF generation requires weasyprint package. Please install it with: pip install weasyprint")

        try:
            # Convert HTML to PDF using weasyprint
            html_doc = weasyprint.HTML(filename=str(html_path))
            
            # Enhanced CSS for better PDF rendering - same as regular report
            css = weasyprint.CSS(string="""
                @page {
                    margin: 1in;
                    size: letter;
                }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    font-size: 11pt;
                }
                h1 {
                    color: #2c3e50;
                    font-size: 24pt;
                    margin-top: 0;
                    page-break-before: avoid;
                }
                h2 {
                    color: #2c3e50;
                    font-size: 18pt;
                    margin-top: 20pt;
                    margin-bottom: 10pt;
                    page-break-after: avoid;
                }
                h3 {
                    color: #2c3e50;
                    font-size: 14pt;
                    margin-top: 15pt;
                    margin-bottom: 8pt;
                    page-break-after: avoid;
                }
                h4 {
                    color: #2c3e50;
                    font-size: 12pt;
                    margin-top: 12pt;
                    margin-bottom: 6pt;
                }
                p {
                    margin-bottom: 8pt;
                    text-align: justify;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 12pt 0;
                    font-size: 10pt;
                    page-break-inside: avoid;
                }
                th, td {
                    border: 1pt solid #ddd;
                    padding: 6pt;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .supported-true {
                    color: green;
                    font-weight: bold;
                }
                .supported-false {
                    color: red;
                    font-weight: bold;
                }
                ul, ol {
                    margin: 8pt 0;
                    padding-left: 20pt;
                }
                li {
                    margin-bottom: 4pt;
                }
            """)
            
            # Generate the PDF
            html_doc.write_pdf(str(output_path), stylesheets=[css])
            
            # Clean up the temporary HTML file
            os.remove(html_path)
            
            self.logger.info(f"Successfully generated PDF summary report at {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF summary report: {e}")
            # Clean up the temporary HTML file if it exists
            if html_path.exists():
                os.remove(html_path)
            
            # Fallback: rename HTML file to indicate it's the fallback
            fallback_path = output_path.parent / f"{output_path.stem}_fallback.html"
            if html_path.exists():
                os.rename(html_path, fallback_path)
                self.logger.warning(f"PDF generation failed, HTML fallback saved at {fallback_path}")
                return str(fallback_path)
            else:
                raise Exception(f"PDF summary generation failed and no fallback available: {e}")
        
    async def _parse_question_robustly(self, raw_text: str) -> Dict[str, str]:
        """
        Parse research question text into structured fields. Uses rule-based first,
        falls back to LLM if structure is ambiguous.
        """
        # Rule-based parsing
        q_content = raw_text.strip()
        rationale_content = ""
        procedure_content = ""
        title_content = ""

        if "RATIONALE:" in raw_text or "QUESTION:" in raw_text.upper():
            import re
            q_match = re.search(r'QUESTION:?\s*(.+?)(?=RATIONALE:|PROCEDURE:|TITLE:|\Z)', raw_text, re.DOTALL | re.IGNORECASE)
            if q_match:
                q_content = q_match.group(1).strip()
            rationale_match = re.search(r'RATIONALE:?\s*(.+?)(?=PROCEDURE:|TITLE:|\Z)', raw_text, re.DOTALL | re.IGNORECASE)
            if rationale_match:
                rationale_content = rationale_match.group(1).strip()
            procedure_match = re.search(r'PROCEDURE:?\s*(.+?)(?=TITLE:|\Z)', raw_text, re.DOTALL | re.IGNORECASE)
            if procedure_match:
                procedure_content = procedure_match.group(1).strip()
            title_match = re.search(r'TITLE:?\s*(.+?)(?=\Z)', raw_text, re.DOTALL | re.IGNORECASE)
            if title_match:
                title_content = title_match.group(1).strip()

        # If rule-based left question empty or unclear, use LLM
        if (not q_content or len(q_content) < 10) and self.llm_interface and "question_parser" in getattr(self, "reporter_prompts", {}):
            try:
                prompt_config = self.reporter_prompts["question_parser"]
                user_prompt = prompt_config.get("user_prompt", "{raw_text}").format(raw_text=raw_text[:3000])
                response = await self.llm_interface.generate(
                    prompt=user_prompt,
                    system_message=prompt_config.get("system_prompt", "Parse the question."),
                    agent_name="reporter"
                )
                if response:
                    import json
                    raw = response.strip()
                    if "```" in raw:
                        import re
                        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                        if m:
                            raw = m.group(1).strip()
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = {"question": raw_text, "rationale": "", "procedure": "", "title": ""}
                    q_content = parsed.get("question", q_content) or raw_text
                    rationale_content = parsed.get("rationale", rationale_content)
                    procedure_content = parsed.get("procedure", procedure_content)
                    title_content = parsed.get("title", title_content)
            except Exception as e:
                self.logger.warning(f"LLM question parsing failed: {e}, using rule-based result")

        if not q_content:
            q_content = raw_text
        return {"question": q_content, "rationale": rationale_content, "procedure": procedure_content, "title": title_content}

    # Exclude: Python keywords, built-ins, loop vars, and common non-output names.
    # No allowlist—extraction is generalizable to any interpretability analysis.
    _ANALYSIS_VAR_EXCLUDE = frozenset({
        "if", "elif", "else", "for", "while", "with", "def", "class", "return", "try",
        "except", "finally", "and", "or", "not", "in", "is", "lambda", "yield",
        "type", "id", "format", "str", "int", "float", "list", "dict", "set", "tuple",
        "len", "range", "enumerate", "zip", "map", "filter", "open", "print",
        "i", "j", "k", "n", "m", "x", "y", "z", "f", "e", "p", "v", "c",
        "tmp", "temp", "val", "item", "key", "idx", "_", "self", "cls",
    })

    def _extract_analysis_variables(self, code: str) -> List[str]:
        """
        Extract variable names from analysis code using exclude-only heuristics.
        Generalizable to any interpretability analysis (causal tracing, attention, SAEs, etc.).
        """
        import re
        seen = set()
        vars_found = []

        # Assignment targets: name = ...
        for m in re.finditer(r'^\s*(\w+)\s*=', code, re.MULTILINE):
            name = m.group(1).lower()
            if name in self._ANALYSIS_VAR_EXCLUDE or name in seen:
                continue
            if len(name) == 1 and name != "_":
                continue
            seen.add(name)
            vars_found.append((name, self._variable_relevance_score(name)))

        # Also catch "for x in" style (we exclude single-char in exclude, but multi-char is useful)
        for m in re.finditer(r'\bfor\s+(\w+)\s+in\s+', code):
            name = m.group(1).lower()
            if name not in self._ANALYSIS_VAR_EXCLUDE and name not in seen and len(name) > 1:
                seen.add(name)
                vars_found.append((name, 0))  # Lower relevance for loop vars

        # Sort by relevance (higher first), then dedupe and take top N
        vars_found.sort(key=lambda x: -x[1])
        result = list(dict.fromkeys(v[0] for v in vars_found))
        return result[:35]

    def _variable_relevance_score(self, name: str) -> int:
        """Score variable by how likely it is to hold analysis output (higher = more relevant)."""
        score = 0
        name_lower = name.lower()
        # Output-like suffixes
        if any(name_lower.endswith(s) for s in ("results", "summary", "output", "matrix", "profile", "scores", "weights")):
            score += 3
        if any(s in name_lower for s in ("cache", "activation", "logits", "embedding")):
            score += 2
        if any(s in name_lower for s in ("model", "df", "data", "fig", "ax")):
            score += 2
        # Longer names often more substantive
        if len(name) >= 8:
            score += 1
        return score

    async def _extract_visualization_variables_llm(self, analysis_code: str) -> Optional[List[str]]:
        """
        Use LLM to extract variable names a visualization would need from analysis code.
        Generalizable to any interpretability analysis type.
        """
        section_key = "notebook_variable_extractor"
        if not self.llm_interface or not getattr(self, "reporter_prompts", None) or section_key not in self.reporter_prompts:
            return None
        try:
            prompt_config = self.reporter_prompts[section_key]
            user_prompt = prompt_config.get("user_prompt", "").format(
                analysis_code=analysis_code[:8000]
            )
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=prompt_config.get("system_prompt", "Extract variable names."),
                agent_name="reporter"
            )
            if not response or not response.strip():
                return None
            # Parse comma-separated list
            raw = response.strip()
            if "```" in raw:
                m = re.search(r"```(?:[\w]*)\s*([\s\S]*?)```", raw)
                if m:
                    raw = m.group(1).strip()
            vars_list = [v.strip().lower() for v in raw.split(",") if v.strip()]
            vars_list = [v for v in vars_list if v not in self._ANALYSIS_VAR_EXCLUDE]
            return vars_list[:40] if vars_list else None
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.debug(f"LLM variable extraction failed: {e}")
            return None

    async def generate_jupyter_notebook(self, 
                                        question: Union[str, Dict[str, Any]],
                                        analysis_results: Dict[str, Any],
                                        evaluation_results: Dict[str, Any],
                                        visualizations: Dict[str, str],
                                        task_config: Optional[Dict[str, Any]] = None,
                                        title: Optional[str] = None,
                                        output_path: Optional[Path] = None) -> Path:
        """
        Generates a self-contained, reproducible Jupyter notebook (.ipynb).
        Run all cells from top to bottom to reproduce the full analysis and visualizations.
        """
        if output_path is None:
            raise ValueError("output_path must be provided")

        print(f"Generating self-contained Jupyter Notebook at {output_path}...")
        use_minimal = False
        for attempt in range(2):
            try:
                path = await self._generate_jupyter_notebook_impl(
                    question=question,
                    analysis_results=analysis_results,
                    evaluation_results=evaluation_results,
                    visualizations=visualizations,
                    task_config=task_config,
                    title=title,
                    output_path=output_path,
                    use_minimal_adaptations=use_minimal,
                )
                validation_ok = self._validate_notebook_runs(path)
                if validation_ok:
                    print(f"[AUTOINTERP] Notebook generation complete and validated at {path}")
                    return path
                if attempt == 0:
                    print("[AUTOINTERP] Notebook validation failed, regenerating with minimal adaptations...")
                    use_minimal = True
                    continue
                print(f"[AUTOINTERP] Notebook saved at {path} (validation reported errors - review before use)")
                return path
            except Exception as e:
                if attempt == 0:
                    print(f"[AUTOINTERP] Notebook generation failed: {e}")
                    print("[AUTOINTERP] Retrying with minimal adaptations...")
                    use_minimal = True
                    continue
                raise
        return output_path  # unreachable

    async def _generate_jupyter_notebook_impl(self,
                                        question: Union[str, Dict[str, Any]],
                                        analysis_results: Dict[str, Any],
                                        evaluation_results: Dict[str, Any],
                                        visualizations: Dict[str, str],
                                        task_config: Optional[Dict[str, Any]] = None,
                                        title: Optional[str] = None,
                                        output_path: Optional[Path] = None,
                                        use_minimal_adaptations: bool = False) -> Path:
        # --- 1. Initialize Notebook & Metadata ---
        nb = nbformat.v4.new_notebook()
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3 (Data Science)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.10"
            }
        }

        # --- 2. How to Run & Environment Guidance ---
        project_dir = str(self.path_resolver.get_project_dir().resolve())
        guidance_md = """## How to Run This Notebook

1. **Environment**: Python 3.10+ recommended. CUDA-capable GPU recommended for transformer models.
2. **First run**: Execute the setup cell to install packages and load imports. Comment out the `pip install` line after the first run to speed up subsequent runs.
3. **Execution**: Run all cells from top to bottom (Cell → Run All). Do not skip cells; later cells depend on variables from earlier ones.
4. **Paths**: The notebook sets `PROJECT_DIR` and changes to the project directory. Place this notebook in the project's `reports/` folder for paths to resolve correctly.
5. **Output**: All visualizations display inline. No files are saved from this notebook.
"""
        nb.cells.append(nbformat.v4.new_markdown_cell(guidance_md))

        # --- 3. Markdown Header & Intro ---
        raw_text = question.get("text", str(question)) if isinstance(question, dict) else str(question)
        try:
            parsed = await self._parse_question_robustly(raw_text)
        except Exception as e:
            self.logger.warning(f"Question parsing failed: {e}, using raw text")
            parsed = {"question": raw_text, "rationale": "", "procedure": "", "title": "AutoInterp Analysis Report"}
        display_title = title or parsed.get("title", "").strip() or "AutoInterp Analysis Report"
        q_content = parsed["question"]
        rationale_content = parsed["rationale"]
        procedure_content = parsed["procedure"]

        nb.cells.append(nbformat.v4.new_markdown_cell(
            f"# {display_title}\n\n"
            "This notebook is **fully self-contained**. Run all cells from top to bottom to reproduce "
            "the analysis and visualizations. No external scripts are required."
        ))
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "## Hypothesis\n\n"
            "**Research Question:**\n\n"
            f"{q_content}"
        ))
        if rationale_content:
            nb.cells.append(nbformat.v4.new_markdown_cell(
                f"### Rationale\n{rationale_content}"
            ))
        if procedure_content:
            nb.cells.append(nbformat.v4.new_markdown_cell(
                f"### Proposed Procedure\n{procedure_content}"
            ))
        
        # --- 4. Path Setup (reproducibility) ---
        path_setup_code = f'''# Set project directory for consistent paths (required for reproducibility)
PROJECT_DIR = r"{project_dir}"
import os
os.chdir(PROJECT_DIR)
print(f"Working directory: {{os.getcwd()}}")
'''
        path_setup_md = """## Path Setup

**What:** Sets `PROJECT_DIR` to the project root and changes the working directory there so that relative paths in the analysis code (e.g. `data/prompts.csv`) resolve correctly.

**Why:** The analysis scripts were written to assume they run from the project root. Running the notebook from a different directory would break path resolution. Setting `os.chdir(PROJECT_DIR)` ensures consistency with the original pipeline execution.
"""
        nb.cells.append(nbformat.v4.new_markdown_cell(path_setup_md))
        nb.cells.append(nbformat.v4.new_code_cell(path_setup_code))

        # Experiment Setup section with configuration
        setup_md = (
            "## Experiment Setup\n\n"
            "**What:** Installs required packages and loads all imports needed for this notebook. One unified import cell at the top; no duplicate imports in later cells.\n\n"
            "**Why:** The analysis and visualization code depend on these packages. Run this cell once; comment out the `pip install` line after the first run to speed up subsequent runs.\n\n"
            "**Configuration:**"
        )
        if task_config:
            model_cfg = task_config.get("model") or {}
            if model_cfg:
                setup_md += f"\n- **Model:** `{model_cfg.get('name', 'Unknown')}`"
            if "dataset" in task_config:
                setup_md += f"\n- **Dataset:** `{task_config.get('dataset', 'Unknown')}`"
        nb.cells.append(nbformat.v4.new_markdown_cell(setup_md))
        
        # --- 5. Placeholder for Dynamic Setup (will be filled after we collect all code) ---
        setup_cell_index = len(nb.cells)
        nb.cells.append(new_code_cell("# Setup cell - populated below"))
        
        # --- 6. Process Analysis Steps (only successful, sorted by analysis number) ---
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "## Analysis Steps\n\n"
            "The following cells replicate the analysis. Each step is split by function/class. "
            "Run all cells in order; later visualization cells use variables from these steps."
        ))
        analyses = analysis_results.get("analyses", [])
        # Filter to steps that executed successfully (executor uses "success"; some may use "status")
        def _is_success(step: Dict[str, Any]) -> bool:
            if step.get("success") is False:
                return False
            if step.get("success") is True:
                return True
            return step.get("status") == "success"
        successful_steps = [a for a in analyses if _is_success(a)]
        if not successful_steps:
            nb.cells.append(nbformat.v4.new_markdown_cell(
                "**No successful analysis scripts were found.**\n\n"
                "All analyses failed execution or no analyses were run. Re-run the pipeline to generate "
                "successful analyses before this notebook can reproduce the results."
            ))
        # Sort by analysis number so order is consistent (analysis_1, analysis_2, ...)
        def _sort_key(s):
            n = self._extract_analysis_number_from_step(s)
            return (n if n is not None else 999,)
        successful_steps = sorted(successful_steps, key=_sort_key)
        # Include only the FINAL successful analysis to avoid redundant steps. Each iteration
        # refines the same question; the last one has the most complete, correct implementation.
        if len(successful_steps) > 1:
            successful_steps = [successful_steps[-1]]
        
        all_notebook_code = []
        for i, step in enumerate(successful_steps, 1):
            # Load code from step or script_path; fallback to final attempt from disk
            raw_code = self._get_analysis_code(step, step_index=i)
            if not raw_code.strip():
                continue

            # A. Sanitize (Fix null/true/false artifacts)
            clean_code = self._sanitize_code(raw_code)

            # B. Generate Explanation (Rationale + Alternatives)
            raw_result = step.get("results", "")
            result_output = raw_result.get("stdout", "") if isinstance(raw_result, dict) else str(raw_result)
            try:
                explanation_md = await self._generate_step_explanation(i, clean_code, result_output)
            except Exception as e:
                self.logger.warning(f"Step explanation failed for step {i}: {e}")
                explanation_md = f"## Step {i}: Analysis Execution\n\n*Explanation generation failed: {e}*"
            nb.cells.append(nbformat.v4.new_markdown_cell(explanation_md))

            # C. Modularize into chunks, adapt each for notebook (skip import chunks - single import cell at top)
            code_chunks = self._split_code_into_logical_chunks(clean_code)
            previous_vars_in_step = []

            for chunk in code_chunks:
                if chunk["title"] == "Imports & Dependencies":
                    all_notebook_code.append(chunk["content"])
                    continue
                try:
                    desc = await self._generate_chunk_description(chunk["content"], chunk["title"])
                except Exception as e:
                    self.logger.warning(f"Chunk description failed: {e}")
                    desc = chunk["title"]
                chunk_md = f"**{chunk['title']}**\n\n{desc}"
                nb.cells.append(nbformat.v4.new_markdown_cell(chunk_md))

                if use_minimal_adaptations:
                    content = self._adapt_chunk_minimal(chunk["content"])
                else:
                    content = await self._adapt_chunk_for_notebook(
                        chunk_content=chunk["content"],
                        chunk_title=chunk["title"],
                        step_index=i,
                        previous_vars=previous_vars_in_step,
                    )
                if not content:
                    content = self._adapt_chunk_minimal(chunk["content"]) or chunk["content"]
                if content.strip():
                    header_comment = f"# Step {i}: {chunk['title']}\n"
                    content = header_comment + content
                    all_notebook_code.append(content)
                nb.cells.append(nbformat.v4.new_code_cell(content))
                new_vars = self._extract_analysis_variables(content)
                previous_vars_in_step = list(dict.fromkeys(previous_vars_in_step + new_vars))
            
            # D. Output Summary (filter out file-save lines; notebook stores results in variables)
            if result_output:
                filtered_output = self._filter_verifiable_output_for_notebook(result_output)
                nb.cells.append(nbformat.v4.new_markdown_cell(f"**Verifiable Output:**\n```text\n{filtered_output}\n```"))

            # E. Visualization for this analysis step - LLM adapts for notebook (use variables, display inline)
            analysis_name = self._extract_analysis_name(step, i)
            if analysis_name in visualizations:
                viz_code = self._load_visualization_code(analysis_name)
                if viz_code:
                    clean_viz_raw = self._sanitize_code(viz_code)
                    analysis_code_for_step = clean_code
                    if use_minimal_adaptations:
                        adapted_viz = self._adapt_viz_minimal(clean_viz_raw)
                    else:
                        try:
                            available_vars = await self._extract_visualization_variables_llm(analysis_code_for_step)
                        except Exception:
                            available_vars = self._extract_analysis_variables(analysis_code_for_step)
                        if not available_vars:
                            available_vars = self._extract_analysis_variables(analysis_code_for_step)
                        adapted_viz = await self._adapt_viz_for_notebook(
                            analysis_code_summary=analysis_code_for_step[:4000],
                            available_variables=available_vars,
                            original_viz_code=clean_viz_raw
                        )
                        if not adapted_viz:
                            adapted_viz = self._adapt_viz_minimal(clean_viz_raw)
                            self.logger.info(f"Using minimal viz adaptation for step {i} (LLM failed)")
                    if adapted_viz:
                        all_notebook_code.append(adapted_viz)
                        if "plt.show()" not in adapted_viz and "fig.show()" not in adapted_viz and "display(" not in adapted_viz:
                            adapted_viz = adapted_viz.rstrip() + "\n\nplt.show()  # Display figure inline"
                        try:
                            viz_explanation = await self._generate_viz_explanation(
                                step_index=i,
                                result_excerpt=result_output[:1500],
                                viz_code_excerpt=adapted_viz[:2000],
                            )
                        except Exception as e:
                            self.logger.warning(f"Viz explanation failed: {e}")
                            viz_explanation = "*Visualization for this analysis step. Uses variables from above.*"
                        nb.cells.append(nbformat.v4.new_markdown_cell(
                            f"### Visualization for Step {i}\n\n{viz_explanation}\n\n"
                            f"*Uses variables from the analysis cells above. Displays inline (no file save).*"
                        ))
                        nb.cells.append(nbformat.v4.new_code_cell(adapted_viz))
                    else:
                        err_msg = getattr(self, '_last_viz_error', 'Unknown error')
                        nb.cells.append(nbformat.v4.new_markdown_cell(
                            f"### Visualization for Step {i}\n\n"
                            f"*Visualization adaptation failed: {err_msg}*"
                        ))
                else:
                    nb.cells.append(nbformat.v4.new_markdown_cell(
                        f"### Visualization for Step {i}\n\n"
                        f"*Visualization script not found on disk. Re-run the pipeline to regenerate.*"
                    ))

        # --- 7. Replace placeholder with unified setup cell (all imports from entire notebook) ---
        combined_code = "\n\n".join(all_notebook_code)
        unified_setup_cell = self._create_setup_cell_from_code(combined_code)
        nb.cells[setup_cell_index] = unified_setup_cell

        # --- 8. Evaluation & Conclusion ---
        conclusion = evaluation_results.get("conclusion", "Analysis Complete")
        if isinstance(conclusion, dict): conclusion = str(conclusion)
            
        eval_md = f"""## Evaluation and Conclusion
        
### Final Conclusion
**{conclusion.upper()}**

### Reasoning
{evaluation_results.get('reasoning', 'See full report for detailed reasoning.')}
"""
        nb.cells.append(nbformat.v4.new_markdown_cell(eval_md))
        
        # --- 9. Save File ---
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        return output_path

    def _validate_notebook_runs(self, nb_path: Path, timeout: int = 300) -> bool:
        """
        Execute the notebook to validate it runs. Returns True if execution succeeds.
        """
        import subprocess
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_name = "validated.ipynb"
                cwd = str(self.path_resolver.get_project_dir())
                cmd = [
                    sys.executable, "-m", "nbconvert",
                    "--ExecutePreprocessor.timeout=" + str(timeout),
                    "--to", "notebook",
                    "--execute",
                    str(nb_path),
                    "--output", out_name,
                    "--output-dir", tmpdir,
                ]
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 60,
                )
            if result.returncode != 0:
                self.logger.warning(f"Notebook validation failed: {result.stderr or result.stdout}")
                return False
            return True
        except subprocess.TimeoutExpired:
            self.logger.warning("Notebook validation timed out")
            return False
        except (FileNotFoundError, ModuleNotFoundError):
            self.logger.debug("nbconvert not found, skipping notebook validation")
            return True  # Don't fail if nbconvert not installed
        except Exception as e:
            self.logger.warning(f"Notebook validation error: {e}")
            return False

    # --- HELPER METHODS (Ensure these are in the class) ---

    async def _adapt_chunk_for_notebook(
        self,
        chunk_content: str,
        chunk_title: str,
        step_index: int,
        previous_vars: List[str],
    ) -> Optional[str]:
        """
        Rewrite a code chunk for notebook context: inline comments, PROJECT_DIR paths, no prompt saves.
        Retries once on failure before returning None.
        """
        section_key = "notebook_chunk_adapter"
        if not self.llm_interface or not hasattr(self, "reporter_prompts") or section_key not in self.reporter_prompts:
            return None
        for attempt in range(2):
            try:
                prompt_config = self.reporter_prompts[section_key]
                user_prompt = prompt_config.get("user_prompt", "").format(
                    step_index=step_index,
                    chunk_title=chunk_title,
                    previous_vars=", ".join(previous_vars) if previous_vars else "(none yet)",
                    chunk_content=chunk_content[:8000],
                )
                response = await self.llm_interface.generate(
                    prompt=user_prompt,
                    system_message=prompt_config.get("system_prompt", "Adapt chunk for notebook."),
                    agent_name="reporter",
                )
                if not response or not response.strip():
                    if attempt == 0:
                        continue
                    return None
                code = response.strip()
                if "```" in code:
                    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", code)
                    if m:
                        code = m.group(1).strip()
                return self._sanitize_code(code)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"Chunk adaptation failed (attempt {attempt + 1}/2): {e}")
                if attempt == 0:
                    continue
                return None
        return None

    async def _generate_viz_explanation(
        self,
        step_index: int,
        result_excerpt: str,
        viz_code_excerpt: str,
    ) -> str:
        """Generate what/why explanation for a visualization cell."""
        section_key = "notebook_viz_explanation"
        if not self.llm_interface or not hasattr(self, "reporter_prompts") or section_key not in self.reporter_prompts:
            return "*Explanation unavailable (prompt not found).*"
        try:
            prompt_config = self.reporter_prompts[section_key]
            user_prompt = prompt_config.get("user_prompt", "").format(
                step_index=step_index,
                result_excerpt=result_excerpt[:1500],
                viz_code_excerpt=viz_code_excerpt[:2000],
            )
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=prompt_config.get("system_prompt", "Explain the visualization."),
                agent_name="reporter",
            )
            return response.strip() if response else "*Explanation generation failed.*"
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Viz explanation failed: {e}")
            return f"*Explanation failed: {type(e).__name__}: {str(e)}*"

    async def _adapt_viz_for_notebook(
        self,
        analysis_code_summary: str,
        available_variables: List[str],
        original_viz_code: str
    ) -> Optional[str]:
        """
        Use LLM to adapt standalone viz script for notebook: use analysis variables, display inline, no saves.
        Retries once on failure before returning None.
        """
        self._last_viz_error = None
        section_key = "notebook_viz_adapter"
        if not self.llm_interface or not hasattr(self, "reporter_prompts") or section_key not in self.reporter_prompts:
            self._last_viz_error = "Prompt 'notebook_viz_adapter' not found"
            return None
        for attempt in range(2):
            try:
                prompt_config = self.reporter_prompts[section_key]
                user_prompt = prompt_config.get("user_prompt", "").format(
                    analysis_code_summary=analysis_code_summary,
                    available_variables=", ".join(available_variables) if available_variables else "(infer from analysis code)",
                    original_viz_code=original_viz_code[:6000]
                )
                response = await self.llm_interface.generate(
                    prompt=user_prompt,
                    system_message=prompt_config.get("system_prompt", "Adapt visualization for notebook."),
                    agent_name="reporter"
                )
                if not response or not response.strip():
                    self._last_viz_error = "LLM returned empty response"
                    if attempt == 0:
                        continue
                    return None
                code = response.strip()
                if "```" in code:
                    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", code)
                    if m:
                        code = m.group(1).strip()
                return self._sanitize_code(code)
            except Exception as e:
                self._last_viz_error = str(e)
                if hasattr(self, "logger"):
                    self.logger.warning(f"Viz adaptation failed (attempt {attempt + 1}/2): {e}")
                if attempt == 0:
                    continue
                return None
        return None

    def _adapt_viz_minimal(self, viz_code: str) -> str:
        """
        Minimal adaptation of viz code for notebook: remove savefig/write, add show. No LLM.
        Used as fallback when LLM adaptation fails.
        """
        code = self._sanitize_code(viz_code)
        lines = code.split('\n')
        out = []
        for line in lines:
            if re.search(r'\.(savefig|write_image|write_html)\s*\(', line):
                continue
            out.append(line)
        code = '\n'.join(out).rstrip()
        if 'plt.show()' not in code and 'fig.show()' not in code and 'display(' not in code:
            code += '\n\nplt.show()  # Display inline'
        return code

    def _adapt_chunk_minimal(self, chunk_content: str) -> str:
        """
        Minimal adaptation: sanitize only. No LLM; used when chunk adaptation fails.
        Path setup is already done via PROJECT_DIR in an earlier cell.
        """
        return self._sanitize_code(chunk_content)

    async def _generate_step_explanation(self, index: int, code: str, result: str) -> str:
        """
        Generates commentary using the prompt defined in prompts/reporter.yaml.
        Uses full code and result; returns specific failure message on error.
        """
        header = f"## Step {index}: Analysis Execution"
        section_key = "notebook_step_explainer"
        
        if not self.llm_interface or not hasattr(self, 'reporter_prompts') or section_key not in self.reporter_prompts:
            return f"{header}\n*Explanation failed: Prompt '{section_key}' not found in reporter prompts.*"

        try:
            prompt_config = self.reporter_prompts[section_key]
            system_prompt = prompt_config.get("system_prompt", "You are a helpful assistant.")
            user_prompt_template = prompt_config.get("user_prompt", "{code_content}")
            user_prompt = user_prompt_template.format(
                step_index=index,
                code_content=code,
                execution_result=result
            )
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=system_prompt,
                agent_name="reporter"
            )
            return f"{header}\n\n{response.strip()}"
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating explanation for step {index}: {e}")
            return f"{header}\n*Explanation generation failed: {type(e).__name__}: {str(e)}*"

    def _filter_verifiable_output_for_notebook(self, result_output: str) -> str:
        """
        Remove lines about saving to disk from displayed output. The notebook stores results
        in variables instead of files, so such lines would be misleading.
        """
        import re
        lines = result_output.split("\n")
        filtered = []
        for line in lines:
            # Skip lines that mention saving results to a file
            if re.search(r"Saved\s+\w+\s+to\s+|saved\s+to\s+|Results\s+saved\s+to", line, re.IGNORECASE):
                continue
            if re.search(r"Writing\s+to\s+.*\.(json|csv|png|pdf)", line, re.IGNORECASE):
                continue
            filtered.append(line)
        return "\n".join(filtered).rstrip()

    def _sanitize_code(self, code: str) -> str:
        """
        Cleans up common JSON-to-Python artifacts and format issues.
        Avoids replacing null/true/false inside string literals to prevent breaking valid code.
        """
        import re
        # Remove markdown code fences only at boundaries
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:].lstrip()
        elif code.startswith("```"):
            code = code[3:].lstrip()
        if code.endswith("```"):
            code = code[:-3].rstrip()
        # Fix JSON booleans/nulls only when NOT inside quotes (simplified: only at word boundaries
        # and not preceded by quote). Use a conservative approach: replace only in contexts
        # that look like Python (e.g., after = or , or () with optional whitespace).
        def replace_json_token(match: re.Match) -> str:
            token = match.group(0).lower()
            if token == "null":
                return "None"
            if token == "true":
                return "True"
            if token == "false":
                return "False"
            return match.group(0)
        # Only replace when it looks like a standalone token (not in string)
        # Pattern: word boundary, then null/true/false, then word boundary
        code = re.sub(r'(?<=[=\s,(\[])\s*(null|true|false)\s*(?=[\s,)\];])', replace_json_token, code, flags=re.IGNORECASE)
        # Ensure consistent spacing around def/class
        code = re.sub(r'\n+(def|class) ', r'\n\n\1 ', code)
        return code.strip()

    def _split_code_into_logical_chunks(self, code: str) -> List[Dict[str, str]]:
        """
        Splits script by function and class boundaries only. Never splits mid-statement.
        Each chunk is a complete def, class, or top-level block.
        """
        import re
        chunks = []
        lines = code.split('\n')

        # Extract Imports (standalone)
        import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]
        other_lines = [l for l in lines if not l.strip().startswith(('import ', 'from '))]

        if import_lines:
            chunks.append({"type": "code", "title": "Imports & Dependencies", "content": "\n".join(import_lines)})

        remaining_code = "\n".join(other_lines).strip()
        if not remaining_code:
            return chunks

        # Split by def, class, or if __name__ - keep each block intact
        # Match: start of line (or after newline), then def/class/if __name__
        pattern = r'\n(?=(?:def |class |if __name__\s*==))'
        raw_sections = re.split(pattern, remaining_code)

        # First section might not start with def/class (e.g. top-level config)
        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            title = "Analysis Logic"
            if section.startswith("def "):
                name = section.split('(')[0].replace("def ", "").strip()
                title = f"Define Helper: `{name}`"
            elif section.startswith("class "):
                name = section.split(':')[0].replace("class ", "").strip()
                title = f"Define Class: `{name}`"
            elif section.startswith("if __name__"):
                title = "Main Execution Block"
            elif section.startswith("#"):
                title = section.split('\n')[0].replace("#", "").strip()
            elif "plt." in section or "fig." in section or "sns." in section:
                title = "Visualization"

            chunks.append({"type": "code", "title": title, "content": section})

        return chunks

    async def _generate_chunk_description(self, chunk_content: str, title: str) -> str:
        """
        Generate a brief description for a code chunk using the LLM.
        Returns specific failure message on error.
        """
        section_key = "notebook_chunk_descriptor"
        if not self.llm_interface or not hasattr(self, "reporter_prompts") or section_key not in self.reporter_prompts:
            if chunk_content.strip().startswith("#"):
                return chunk_content.split("\n")[0].replace("#", "").strip()
            return title

        try:
            prompt_config = self.reporter_prompts[section_key]
            system_prompt = prompt_config.get("system_prompt", "Describe this code briefly.")
            user_prompt_template = prompt_config.get("user_prompt", "{chunk_content}")
            user_prompt = user_prompt_template.format(chunk_content=chunk_content[:6000])
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=system_prompt,
                agent_name="reporter",
            )
            return (response.strip()[:3000] if response else title)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Chunk description generation failed: {e}")
            return f"*Chunk description failed: {type(e).__name__}: {str(e)}*"

    def _extract_analysis_number_from_step(self, step: Dict[str, Any]) -> Optional[int]:
        """Extract analysis number from step (e.g. 3 from script_path .../analysis_3/...)."""
        script_path = step.get("script_path", "")
        if script_path:
            match = re.search(r"analysis_(\d+)", str(script_path))
            if match:
                return int(match.group(1))
        return None

    def _extract_analysis_name(self, step: Dict[str, Any], step_index: int) -> str:
        """Extract analysis name (e.g. 'analysis_1') from step for viz lookup."""
        num = self._extract_analysis_number_from_step(step)
        return f"analysis_{num}" if num is not None else f"analysis_{step_index}"

    def _get_analysis_code(self, step: Dict[str, Any], step_index: Optional[int] = None) -> str:
        """
        Get analysis code from step. Prefers script_path (the actual executed script).
        Falls back to resolving the final attempt from disk. Uses analysis number from
        step (not loop index) for correct disk lookup when steps are filtered.
        """
        code = step.get("code", "").strip()
        if code:
            return code
        script_path = step.get("script_path", "")
        if script_path and Path(script_path).exists():
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Could not read analysis script at {script_path}: {e}")
        analysis_num = self._extract_analysis_number_from_step(step)
        if analysis_num is not None:
            return self._resolve_final_script_from_disk(analysis_num)
        if step_index is not None:
            return self._resolve_final_script_from_disk(step_index)
        return ""

    def _resolve_final_script_from_disk(self, analysis_num: int) -> str:
        """
        Resolve the final (highest) attempt script from the analysis_scripts directory.
        Uses analysis_{n}/attempt_{m}/analysis_*.py - takes highest attempt number.
        """
        try:
            analysis_scripts_dir = self.path_resolver.get_path("analysis_scripts")
            analysis_dir = analysis_scripts_dir / f"analysis_{analysis_num}"
            if not analysis_dir.exists():
                return ""
            attempt_dirs = []
            for item in analysis_dir.iterdir():
                if item.is_dir() and item.name.startswith("attempt_"):
                    try:
                        attempt_num = int(item.name.split("_")[1])
                        attempt_dirs.append((attempt_num, item))
                    except (ValueError, IndexError):
                        continue
            if not attempt_dirs:
                return ""
            attempt_dirs.sort(key=lambda x: x[0])
            highest_attempt_dir = attempt_dirs[-1][1]
            # Find analysis_*.py (executed script) - not analysis_generator_*.txt
            for f in highest_attempt_dir.iterdir():
                if f.is_file() and f.name.startswith("analysis_") and f.suffix == ".py":
                    with open(f, "r", encoding="utf-8") as fp:
                        return fp.read()
        except Exception as e:
            self.logger.warning(f"Could not resolve final script from disk for analysis_{analysis_num}: {e}")
        return ""

    def _load_visualization_code(self, viz_name: str) -> Optional[str]:
        """
        Load the FINAL successful visualization Python code from disk. Prefers
        retry scripts (later attempts) over initial attempts. Skips files that
        contain only a plan (markdown) instead of executable Python.
        """
        viz_dir = self.path_resolver.ensure_path("visualizations")
        if not viz_dir.exists():
            return None
        pattern = f"visualization_{viz_name}_*.py"
        all_matches = list(viz_dir.glob(pattern))
        if not all_matches:
            return None

        def sort_key(p: Path) -> tuple:
            """Prefer retry scripts (later attempts); then by mtime. Descending so retries come first."""
            name = p.name
            retry_match = re.search(r"_retry(\d+)\.py$", name)
            retry_num = int(retry_match.group(1)) if retry_match else 0
            has_retry = 1 if "_retry" in name else 0
            return (has_retry, retry_num, -p.stat().st_mtime)

        def is_executable_python(content: str) -> bool:
            """True if content looks like Python code, not a plan/markdown."""
            if not content or len(content.strip()) < 50:
                return False
            # Plan files are markdown (start with **) and lack Python plotting code
            stripped = content.strip()
            if stripped.startswith("**") and "import " not in content:
                return False
            return "import " in content or "plt." in content or "fig." in content or "ax." in content

        matches = sorted(all_matches, key=sort_key, reverse=True)  # Retries first
        for p in matches:
            try:
                content = p.read_text(encoding="utf-8")
                if is_executable_python(content):
                    return content
            except Exception as e:
                self.logger.warning(f"Could not read visualization script {p}: {e}")
        return None

    def _get_imports_and_requirements_from_code(self, all_code: str) -> Tuple[List[str], List[str]]:
        """Extract import lines and pip requirements from code. Returns (import_lines, pip_packages)."""
        import re
        IMPORT_TO_PIP = {
            "sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "Pillow",
            "yaml": "PyYAML", "transformer_lens": "transformer_lens",
            "plotly": "plotly", "wandb": "wandb", "jaxtyping": "jaxtyping"
        }
        # Python stdlib modules - never pip-install these
        STDLIB = {
            "os", "sys", "json", "math", "re", "time", "typing", "pathlib", "collections",
            "random", "logging", "warnings", "datetime", "io", "subprocess", "tempfile",
            "argparse", "functools", "itertools", "copy", "hashlib", "struct", "base64",
        }
        seen_imports = set()
        import_lines = []
        for m in re.finditer(r'^\s*(?:from\s+\S+\s+import\s+.+|import\s+\S+)', all_code, re.MULTILINE):
            line = m.group(0).strip()
            if line and line not in seen_imports:
                seen_imports.add(line)
                import_lines.append(line)
        modules = re.findall(r'^\s*(?:from|import)\s+(\w+)', all_code, re.MULTILINE)
        pip_pkgs = {"transformer_lens", "torch", "einops", "jaxtyping"}
        for mod in modules:
            if mod not in STDLIB:
                pip_pkgs.add(IMPORT_TO_PIP.get(mod, mod))
        return import_lines, sorted(list(pip_pkgs))

    def _create_setup_cell_from_code(self, all_code: str) -> nbformat.NotebookNode:
        """
        Creates a single unified setup cell with all imports needed by the notebook.
        Extracts imports dynamically from the provided code; merges with standard imports.
        """
        import_lines, requirements = self._get_imports_and_requirements_from_code(all_code)
        pip_line = "# Install dependencies (run once; comment out after first run)\n!pip install -q " + " ".join(requirements) + "\n\n"
        base_imports = [
            "import os", "import sys", "import json", "import random", "import logging", "import warnings",
            "warnings.filterwarnings(\"ignore\", \"Can't initialize NVML\")",
            "import numpy as np", "import torch", "import pandas as pd", "import matplotlib.pyplot as plt",
            "import seaborn as sns", "import scipy", "from scipy import stats", "from scipy.stats import pearsonr, spearmanr",
            "import sklearn", "from sklearn import metrics, decomposition", "from tqdm.auto import tqdm", "import einops",
            "import transformer_lens", "from transformer_lens import HookedTransformer", "import transformers",
        ]

        def _module_key(ln):
            parts = ln.strip().replace("import ", "").replace("from ", "").split()[0]
            return parts.split(".")[0]

        seen = {_module_key(ln) for ln in base_imports if "import" in ln or "from " in ln}
        merged = list(base_imports)
        for line in import_lines:
            key = _module_key(line)
            if key not in seen:
                seen.add(key)
                merged.append(line)
        imports_block = "\n".join(merged) + "\n\n"
        runtime = """os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
parameters = {}
"""
        content = pip_line + imports_block + runtime
        return new_code_cell(content)
    
    def _get_filename_from_title(self, title: str) -> str:
        """
        Generate a clean, informative filename based on the report title.
        
        Args:
            title: The report title
            
        Returns:
            Clean filename for the report (without extension)
        """
        if title and title.strip():
            # Create a clean filename from the title
            # Remove common prefixes like "Interpretability Report:"
            clean_title = title.strip()
            if clean_title.lower().startswith("interpretability report:"):
                clean_title = clean_title[24:].strip()
            
            # Sanitize the title for filename use
            sanitized = self._sanitize_filename(clean_title)
            
            # Limit length and ensure it's not empty
            if sanitized and len(sanitized) > 3:
                return sanitized[:60]  # Reasonable filename length
            else:
                # Fallback to study filename
                return self._get_study_filename()
        else:
            # Fallback to study filename  
            return self._get_study_filename()
    
    def _get_study_filename(self) -> str:
        """
        Generate a clean, informative filename based on the study name.
        
        Returns:
            Clean filename for the report (without extension)
        """
        # Get the current project_id which contains the study name
        project_id = self.path_resolver.project_id
        
        # If project_id has the format "study_name_timestamp", extract just the study name
        if "_20" in project_id:  # Assumes timestamps contain "_20" (year 20XX)
            # Split on the last occurrence of a timestamp pattern
            import re
            # Look for the pattern: _YYYY-MM-DDTHH-MM-SS
            timestamp_pattern = r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$'
            match = re.search(timestamp_pattern, project_id)
            if match:
                # Extract everything before the timestamp
                study_name = project_id[:match.start()]
                return study_name
        
        # Fallback: use the whole project_id or a default name
        if project_id and not project_id.startswith("working_project"):
            return project_id
        else:
            # Last resort: try to extract from config title
            project_title = self.config.get("title")
            if project_title:
                # Create a clean filename from the title
                import re
                sanitized = re.sub(r'[^\w\-\.]', '_', project_title).lower()
                sanitized = re.sub(r'_+', '_', sanitized).strip('_')
                return sanitized
            else:
                return "interpretability_report"
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to be safe for all filesystems.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace unsafe characters and spaces with underscore
        sanitized = re.sub(r'[\\/*?:"<>|\s]', "_", filename)
        # Remove extra underscores and clean up
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        # Trim to reasonable length
        sanitized = sanitized[:100]
        return sanitized.lower()
