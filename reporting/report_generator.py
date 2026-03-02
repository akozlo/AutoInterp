"""
Report Generator module for the AutoInterp Agent Framework.
Creates comprehensive and reproducible interpretability reports.
"""

import os
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
        
    async def generate_jupyter_notebook(self, 
                                        question: Union[str, Dict[str, Any]],
                                        analysis_results: Dict[str, Any],
                                        evaluation_results: Dict[str, Any],
                                        visualizations: Dict[str, str],
                                        task_config: Optional[Dict[str, Any]] = None,
                                        title: Optional[str] = None,
                                        output_path: Optional[Path] = None) -> Path:
        """
        Generates a reproducible Jupyter notebook (.ipynb) using nbformat.
        """
        if output_path is None:
            raise ValueError("output_path must be provided")

        print(f"Generating Modular Jupyter Notebook at {output_path}...")
        
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

        # --- 2. Markdown Header & Intro (rich narrative sections) ---
        display_title = title or "AutoInterp Analysis Report"
        raw_text = question.get("text", str(question)) if isinstance(question, dict) else str(question)
        
        # Parse text (Question/Rationale/Procedure extraction logic)
        q_content = raw_text
        rationale_content = ""
        procedure_content = ""
        
        if "RATIONALE:" in raw_text:
            parts = raw_text.split("RATIONALE:")
            q_content = parts[0].replace("Question:", "").strip()
            remaining = parts[1]
            if "PROCEDURE:" in remaining:
                subparts = remaining.split("PROCEDURE:")
                rationale_content = subparts[0].strip()
                procedure_content = subparts[1].split("TITLE:")[0].strip()
            else:
                rationale_content = remaining.strip()

        # Add Header Cells with rich documentation
        nb.cells.append(nbformat.v4.new_markdown_cell(
            f"# {display_title}\n\n"
            "This notebook provides a **self-contained, reproducible** research report. "
            "Run all cells from top to bottom to reproduce the full analysis."
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
        
        # Experiment Setup section with configuration
        setup_md = (
            "## Experiment Setup\n\n"
            "This section loads all dependencies and configures the environment. "
            "**Rationale:** Mechanistic interpretability requires torch, transformer_lens, and related packages. "
            "**Alternatives:** Use a conda environment or Docker for isolation.\n\n"
            "**Configuration:**"
        )
        if task_config:
            if "model" in task_config:
                setup_md += f"\n- **Model:** `{task_config['model'].get('name', 'Unknown')}`"
            if "dataset" in task_config:
                setup_md += f"\n- **Dataset:** `{task_config.get('dataset', 'Unknown')}`"
        nb.cells.append(nbformat.v4.new_markdown_cell(setup_md))
        
        # --- 3. Dynamic Environment Setup ---
        # (Passes analysis results to scan for imports like sklearn/cv2/wandb)
        nb.cells.append(self._create_setup_cell(analysis_results))
        
        # --- 4. Process Analysis Steps (1:1 mapping to executed scripts) ---
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "## Analysis Steps\n\n"
            "The following cells replicate the analysis performed by the AutoInterp agent. "
            "Each step maps 1:1 to the underlying analysis scripts. "
            "**Rationale:** Presenting modular code chunks improves clarity and debuggability. "
            "**Alternatives:** Run steps independently to explore intermediate results."
        ))
        analyses = analysis_results.get("analyses", [])
        successful_steps = [a for a in analyses if a.get("status") == "success"]
        if not successful_steps and analyses:
            successful_steps = analyses  # Fallback to showing everything if no explicit success
        
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
            
            explanation_md = await self._generate_step_explanation(i, clean_code, result_output)
            nb.cells.append(nbformat.v4.new_markdown_cell(explanation_md))

            # C. Modularize into granular chunks with LLM-generated per-chunk descriptions
            code_chunks = self._split_code_into_logical_chunks(clean_code)

            for chunk in code_chunks:
                desc = await self._generate_chunk_description(chunk["content"], chunk["title"])
                chunk_md = f"**{chunk['title']}**\n\n{desc}"
                nb.cells.append(nbformat.v4.new_markdown_cell(chunk_md))

                content = chunk["content"]
                if chunk["title"] != "Imports & Dependencies" and content.strip():
                    header_comment = f"# Step {i}: {chunk['title']}\n"
                    content = header_comment + content
                nb.cells.append(nbformat.v4.new_code_cell(content))
            
            # D. Output Summary
            if result_output:
                trunc_len = 1000
                trunc_out = result_output[:trunc_len] + f"\n... [Output Truncated]" if len(result_output) > trunc_len else result_output
                nb.cells.append(nbformat.v4.new_markdown_cell(f"**Verifiable Output:**\n```text\n{trunc_out}\n```"))

            # E. Visualization for this analysis step (lookup by analysis_name, not step index)
            analysis_name = self._extract_analysis_name(step, i)
            if analysis_name in visualizations:
                viz_code = self._load_visualization_code(analysis_name)
                if viz_code:
                    clean_viz = self._sanitize_code(viz_code)
                    # Prepend note: viz is self-contained, does not assume step numbering
                    preamble = "# Self-contained visualization: reproduces analysis results inline.\n"
                    preamble += "# Does not depend on step numbers or prior cell variable names.\n\n"
                    clean_viz = preamble + clean_viz
                    if "plt.show()" not in clean_viz and "fig.show()" not in clean_viz and "display(" not in clean_viz:
                        clean_viz = clean_viz.rstrip() + "\n\nplt.show()  # Display figure inline"
                    nb.cells.append(nbformat.v4.new_markdown_cell(
                        f"### Visualization for Step {i}\n\n"
                        f"*This visualization corresponds to the analysis above. "
                        f"Run the cell to regenerate the figure (self-contained; no step numbering assumed).*"
                    ))
                    nb.cells.append(nbformat.v4.new_code_cell(clean_viz))
                else:
                    nb.cells.append(nbformat.v4.new_markdown_cell(
                        f"### Visualization for Step {i}\n\n"
                        f"*Visualization script not found. Re-run the pipeline to regenerate.*"
                    ))

        # --- 5. Evaluation & Conclusion ---
        conclusion = evaluation_results.get("conclusion", "Analysis Complete")
        if isinstance(conclusion, dict): conclusion = str(conclusion)
            
        eval_md = f"""## Evaluation and Conclusion
        
### Final Conclusion
**{conclusion.upper()}**

### Reasoning
{evaluation_results.get('reasoning', 'See full report for detailed reasoning.')}
"""
        nb.cells.append(nbformat.v4.new_markdown_cell(eval_md))
        
        # --- 6. Save File ---
        # (Visualizations are interleaved with analysis steps above)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"[AUTOINTERP] Notebook generation complete at {output_path}")
        return output_path

    # --- HELPER METHODS (Ensure these are in the class) ---

    async def _generate_step_explanation(self, index: int, code: str, result: str) -> str:
        """
        Generates commentary using the prompt defined in prompts/reporter.yaml.
        """
        header = f"## Step {index}: Analysis Execution"
        section_key = "notebook_step_explainer"
        
        # 1. Safety Check: Ensure prompts are loaded
        if not self.llm_interface or not hasattr(self, 'reporter_prompts') or section_key not in self.reporter_prompts:
            return f"{header}\n*Automated explanation unavailable (Prompt '{section_key}' not found).*"

        try:
            # 2. Extract configuration from YAML dict
            prompt_config = self.reporter_prompts[section_key]
            system_prompt = prompt_config.get("system_prompt", "You are a helpful assistant.")
            user_prompt_template = prompt_config.get("user_prompt", "{code_content}")
            
            # 3. Format the prompt (Truncate large outputs to save tokens)
            user_prompt = user_prompt_template.format(
                step_index=index,
                code_content=code[:2500], 
                execution_result=result[:1000]
            )
            
            # 4. Generate Response
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=system_prompt,
                agent_name="reporter"
            )
            
            return f"{header}\n\n{response.strip()}"
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating explanation for step {index}: {e}")
            return f"{header}\n*Error generating explanation: {str(e)}*"

    def _sanitize_code(self, code: str) -> str:
        """Cleans up common JSON-to-Python artifacts and format issues."""
        import re
        # Fix JSON booleans/nulls
        replacements = {r'\bnull\b': 'None', r'\btrue\b': 'True', r'\bfalse\b': 'False'}
        for pattern, replacement in replacements.items():
            code = re.sub(pattern, replacement, code, flags=re.IGNORECASE)
        # Fix markdown artifacts
        code = code.replace("```python", "").replace("```", "")
        # Ensure consistent spacing
        code = re.sub(r'\n+(def|class) ', r'\n\n\n\1 ', code)
        return code.strip()

    MAX_CHUNK_LINES = 25  # Split large blocks for finer granularity

    async def _generate_chunk_description(self, chunk_content: str, title: str) -> str:
        """
        Generate a brief description for a code chunk using the LLM.
        Falls back to title if LLM unavailable.
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
            user_prompt = user_prompt_template.format(chunk_content=chunk_content[:2000])
            response = await self.llm_interface.generate(
                prompt=user_prompt,
                system_message=system_prompt,
                agent_name="reporter",
            )
            return response.strip()[:2000] if response else title
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Chunk description generation failed: {e}")
            return title

    def _split_code_into_logical_chunks(self, code: str) -> List[Dict[str, str]]:
        """
        Splits script into granular logical chunks for step-by-step explanation.
        Keeps chunks small (max ~25 lines) and adds descriptive titles.
        """
        import re
        chunks = []
        lines = code.split('\n')

        # Extract Imports
        import_lines = [l for l in lines if l.strip().startswith(('import ', 'from '))]
        other_lines = [l for l in lines if not l.strip().startswith(('import ', 'from '))]

        if import_lines:
            chunks.append({"type": "code", "title": "Imports & Dependencies", "content": "\n".join(import_lines)})

        remaining_code = "\n".join(other_lines).strip()

        # Primary split: def, class, # headers, if __name__
        raw_sections = re.split(r'\n{2,}(?=# |def |class |if __name__)', remaining_code)

        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            title = "Analysis Logic"
            if section.startswith("def "):
                title = f"Define Helper: `{section.split('(')[0].replace('def ', '')}`"
            elif section.startswith("class "):
                title = f"Define Class: `{section.split(':')[0].replace('class ', '')}`"
            elif section.startswith("#"):
                title = section.split('\n')[0].replace("#", "").strip()
            elif "plt." in section or "fig." in section:
                title = "Visualization"

            # Further split large sections for granularity
            section_lines = section.split('\n')
            if len(section_lines) <= self.MAX_CHUNK_LINES:
                chunks.append({"type": "code", "title": title, "content": section})
            else:
                # Split by double newlines or by common logical boundaries
                sub_sections = re.split(r'\n{2,}', section)
                for sub in sub_sections:
                    sub = sub.strip()
                    if not sub:
                        continue
                    sub_lines = sub.split('\n')
                    if len(sub_lines) <= self.MAX_CHUNK_LINES:
                        chunks.append({"type": "code", "title": title, "content": sub})
                    else:
                        for idx in range(0, len(sub_lines), self.MAX_CHUNK_LINES):
                            block = "\n".join(sub_lines[idx : idx + self.MAX_CHUNK_LINES])
                            if block.strip():
                                chunks.append({"type": "code", "title": f"{title} (part)", "content": block})

        return chunks

    def _extract_analysis_name(self, step: Dict[str, Any], step_index: int) -> str:
        """
        Extract analysis name (e.g. 'analysis_1') from step for viz lookup.
        Does NOT assume step_index equals analysis number—derives from script_path.
        Falls back to step_index only when path cannot be parsed.
        """
        script_path = step.get("script_path", "")
        if script_path:
            # Match analysis_N in path (e.g. .../analysis_scripts/analysis_3/attempt_2/...)
            match = re.search(r"analysis_(\d+)", str(script_path))
            if match:
                return f"analysis_{match.group(1)}"
        return f"analysis_{step_index}"

    def _get_analysis_code(self, step: Dict[str, Any], step_index: Optional[int] = None) -> str:
        """
        Get analysis code from step. Prefers script_path (the actual executed script).
        Falls back to resolving the final attempt from disk when script_path is missing
        or the file does not exist (e.g. project moved). Ensures the FINAL attempt
        script is replicated, not failed intermediate attempts.
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
        # Fallback: resolve final attempt from disk (analysis_N/attempt_M/analysis_*.py)
        if step_index is not None:
            return self._resolve_final_script_from_disk(step_index)
        return ""

    def _resolve_final_script_from_disk(self, step_index: int) -> str:
        """
        Resolve the final (highest) attempt script from the analysis_scripts directory.
        Uses analysis_{n}/attempt_{m}/analysis_*.py - takes highest attempt number.
        """
        try:
            analysis_scripts_dir = self.path_resolver.get_path("analysis_scripts")
            analysis_dir = analysis_scripts_dir / f"analysis_{step_index}"
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
            self.logger.warning(f"Could not resolve final script from disk for step {step_index}: {e}")
        return ""

    def _load_visualization_code(self, viz_name: str) -> Optional[str]:
        """
        Load the FINAL successful visualization Python code from disk. Prefers
        retry scripts (later attempts) over initial attempts. Scripts are named
        visualization_{viz_name}_{timestamp}.py or ..._retry{N}.py.
        """
        viz_dir = self.path_resolver.ensure_path("visualizations")
        if not viz_dir.exists():
            return None
        pattern = f"visualization_{viz_name}_*.py"
        all_matches = list(viz_dir.glob(pattern))
        if not all_matches:
            return None

        def sort_key(p: Path) -> tuple:
            """Prefer retry scripts (later attempts); then by mtime."""
            name = p.name
            # Extract retry number: visualization_analysis_1_xxx_retry2.py -> 2
            retry_match = re.search(r"_retry(\d+)\.py$", name)
            retry_num = int(retry_match.group(1)) if retry_match else 0
            # (has_retry, retry_num, -mtime) so retries come first, higher retry first
            has_retry = 1 if "_retry" in name else 0
            return (has_retry, retry_num, -p.stat().st_mtime)

        matches = sorted(all_matches, key=sort_key)
        try:
            with open(matches[0], "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Could not read visualization script {matches[0]}: {e}")
        return None

    def _get_dynamic_requirements(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Scans code to find necessary pip packages."""
        import re
        IMPORT_TO_PIP = {
            "sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "Pillow",
            "yaml": "PyYAML", "transformer_lens": "transformer_lens", 
            "plotly": "plotly", "wandb": "wandb", "jaxtyping": "jaxtyping"
        }
        STDLIB = {"os", "sys", "json", "math", "re", "time", "typing", "pathlib", "numpy", "pandas", "torch"}
        
        analyses = analysis_results.get("analyses", [])
        all_code = "\n".join([self._get_analysis_code(a, step_index=i + 1) for i, a in enumerate(analyses)])
        matches = re.findall(r'^\s*(?:from|import)\s+(\w+)', all_code, re.MULTILINE)
        
        found = {"transformer_lens", "torch", "einops", "jaxtyping"} # Core defaults
        for module in matches:
            if module not in STDLIB:
                found.add(IMPORT_TO_PIP.get(module, module))
        return sorted(list(found))

    def _create_setup_cell(self, analysis_results: Dict[str, Any]) -> nbformat.NotebookNode:
        """
        Creates the environment setup cell with ALL necessary imports for mechanistic
        interpretability. Ensures the notebook is self-contained and runnable.
        """
        requirements = self._get_dynamic_requirements(analysis_results)
        pip_line = f"# !pip install {' '.join(requirements)}  # Uncomment if packages are missing\n\n"
        content = pip_line + NOTEBOOK_STANDARD_IMPORTS
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
