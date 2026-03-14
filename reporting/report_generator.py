"""
Report Generator module for the AutoInterp Agent Framework.
Creates comprehensive and reproducible interpretability reports.
"""

import asyncio
import os
import sys
import json
import re
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import markdown
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    weasyprint = None
    WEASYPRINT_AVAILABLE = False

from ..core.utils import ensure_directory, get_timestamp, load_yaml, PathResolver, get_comprehensive_log_path
from ..core.llm_interface import LLMInterface
from ..core.codex_runner import run_codex_exec


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

        codex_timeout = self.config.get("codex", {}).get("timeout", 1800)
        max_retries = self.config.get("codex", {}).get("max_validation_retries", 3)
        validation_timeout = self.config.get("codex", {}).get("validation_timeout", 300)
        import time
        t0 = time.monotonic()

        print(f"Generating Jupyter Notebook via Codex at {output_path}...")
        path = await self._generate_notebook_via_codex(
            question=question,
            analysis_results=analysis_results,
            evaluation_results=evaluation_results,
            task_config=task_config,
            title=title,
            output_path=output_path,
            timeout=codex_timeout,
            previous_errors=None,
        )

        error_history: List[str] = []
        for attempt in range(max_retries):
            t_val = time.monotonic()
            print(f"[AUTOINTERP] Validating notebook (timeout={validation_timeout}s)...")
            validation_ok, error_msg = self._validate_notebook_runs(path, timeout=validation_timeout)
            elapsed_val = time.monotonic() - t_val
            print(f"[AUTOINTERP] Validation {'passed' if validation_ok else 'failed'} in {elapsed_val:.0f}s")
            if validation_ok:
                print(f"[AUTOINTERP] Notebook generation complete and validated at {path} (total {time.monotonic() - t0:.0f}s)")
                return path

            if attempt < max_retries - 1:
                err = error_msg or "Validation failed"
                error_history.append(err)
                print(f"[AUTOINTERP] Validation failed (attempt {attempt + 1}/{max_retries}), asking Codex to fix...")
                t_codex = time.monotonic()
                path = await self._generate_notebook_via_codex(
                    question=question,
                    analysis_results=analysis_results,
                    evaluation_results=evaluation_results,
                    task_config=task_config,
                    title=title,
                    output_path=output_path,
                    timeout=codex_timeout,
                    previous_errors=error_history,
                )
                print(f"[AUTOINTERP] Codex fix completed in {time.monotonic() - t_codex:.0f}s")
            else:
                print(f"[AUTOINTERP] Notebook saved at {path} (validation failed after {max_retries} fix attempts - review before use)")
                return path

    def _build_codex_initial_prompt(
        self,
        project_dir: Path,
        output_path: Path,
        script_paths_str: str,
        data_path: str,
        parsed: Dict[str, Any],
        raw_text: str,
        conclusion: str,
        reasoning: str,
        model_info: str,
    ) -> str:
        """Build the initial prompt for Codex to create the notebook."""
        return f"""You are in an AutoInterp project at {project_dir}. Your working directory is the project root. You can read any file in the project. Stay within the project directory.

## CRITICAL: Do NOT Execute the Notebook
Do NOT run `nbconvert`, `jupyter nbconvert`, `jupyter execute`, or any command that executes the notebook. The sandbox blocks Jupyter kernel startup (socket creation). Any such attempt will fail. Your ONLY job is to create or edit the .ipynb file. Validation runs externally after you finish.

## Task
Create a self-contained Jupyter notebook that recreates the analysis and visualizations from the working analysis scripts. Write the notebook to: {output_path}

## Working Analysis Scripts
The successful analysis scripts are located at (relative to project root):
{script_paths_str}

Read these scripts. Recreate their logic in the notebook—do not copy-paste verbatim; adapt for notebook context.

## Notebook Requirements
The notebook MUST satisfy ALL of the following:

1. **Recreate analysis and visualization**: Use the working scripts as reference. Adapt the logic for notebook execution (use variables instead of file I/O for intermediate results; no saving to disk except as needed for reproducibility).

2. **Self-contained**: The notebook must run top-to-bottom without external scripts. It may load data from the project (e.g. {data_path}/). Do NOT import from analysis_scripts or load pre-generated plots.

3. **Inline comments**: EVERY code cell MUST have inline comments explaining what each block or significant line does. Do not leave any code cell uncommented.

4. **Markdown before each code cell**: Before every code cell, include a markdown cell that explains:
   - What the code does
   - What choices were made and why
   - What some alternatives are

5. **Visualizations**: Build visualizations from results computed IN the notebook (use variables from earlier cells). Each plot MUST use the correct data variables from the analysis (e.g. layer-wise scores, correlation values). Set appropriate axis limits and scales so the data is visible and interpretable. Add clear axis labels and titles. Do NOT use hardcoded values or load pre-generated plots from disk. Verify each visualization cell would execute correctly with the variables defined in earlier cells.

## Quality Checks (MUST DO Before Delivering)
1. **Fix indentation**: Scan every code cell for Python indentation issues. Ensure all code parses correctly.
2. **Validate visualization code**: For each visualization cell, trace back to ensure the variables (recovery scores, layer indices, etc.) exist and are correctly referenced. Fix any NameError or wrong-variable issues. Do NOT run the notebook to test—just verify the code structure.

## Research Question
{parsed.get('question', raw_text)}

## Rationale
{parsed.get('rationale', '')}

## Procedure
{parsed.get('procedure', '')}

## Evaluation Conclusion
{conclusion}

## Reasoning
{reasoning}
{model_info}

## Deliverable
Create the notebook using nbformat or by writing the .ipynb JSON. After creating it, run quality checks 1 and 2. Deliver once the notebook is complete, well-commented, and visually correct. The file must exist at {output_path} when you are done."""

    def _build_codex_fix_prompt(
        self,
        output_path: Path,
        validation_error: str,
        previous_errors: Optional[List[str]] = None,
    ) -> str:
        """Build the fix prompt when validation fails."""
        error_preview = validation_error[:4000] if len(validation_error) > 4000 else validation_error
        prev_section = ""
        if previous_errors:
            prev_previews = []
            for i, pe in enumerate(previous_errors, 1):
                p = pe[:800] + ("..." if len(pe) > 800 else "")
                prev_previews.append(f"Attempt {i} (your fix did not resolve this):\n```\n{p}\n```")
            prev_section = "\n## Previous Attempts (Do NOT Repeat)\nYou already tried to fix this. The following errors occurred after your prior fix(es). Try a DIFFERENT approach—do not make the same change again.\n\n" + "\n\n".join(prev_previews) + "\n\n"
        return f"""You are in an AutoInterp project. The notebook at {output_path} failed validation when executed (by external validation, not by you).

Do NOT run nbconvert, jupyter nbconvert, or any command to execute the notebook. The sandbox blocks it. Just fix the notebook file and save it.
{prev_section}## Current Validation Error
```
{error_preview}
```

## Your Task
Fix the notebook so it executes successfully from top to bottom. Use a different fix than before if this is a repeat. Common issues:
- **NameError/undefined variable**: A visualization or analysis cell references a variable that doesn't exist or is misspelled. Trace variable definitions from earlier cells.
- **SyntaxError/IndentationError**: Fix Python syntax and indentation.
- **ImportError**: Add missing imports or fix import paths.
- **Visualization errors**: Ensure plot code uses the correct variables (e.g. from the analysis). Check axis labels, data arrays, and that variables exist before use.

Also ensure:
- Every code cell has inline comments.
- Each code cell has a markdown cell above it explaining the code.
- Visualizations use the right data and display meaningful information.

Read the notebook, identify the cause of the error, fix it, and save the corrected notebook to {output_path}."""

    async def _generate_notebook_via_codex(
        self,
        question: Union[str, Dict[str, Any]],
        analysis_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        task_config: Optional[Dict[str, Any]],
        title: Optional[str],
        output_path: Path,
        timeout: int = 1800,
        previous_errors: Optional[List[str]] = None,
    ) -> Path:
        """
        Generate or fix a Jupyter notebook by invoking Codex CLI. Codex stays in
        workspace-write (project dir only). When previous_errors is set, asks Codex
        to fix the existing notebook; includes prior errors so it does not repeat the same fix.
        """
        project_dir = self.path_resolver.get_project_dir().resolve()
        data_path = self.path_resolver.get_path("data")
        if isinstance(data_path, Path):
            data_path = str(data_path)
        else:
            data_path = str(project_dir / "data")

        # Filter to successful analyses
        def _is_success(step: Dict[str, Any]) -> bool:
            if step.get("success") is False:
                return False
            if step.get("success") is True:
                return True
            return step.get("status") == "success"

        analyses = analysis_results.get("analyses", [])
        successful_steps = [a for a in analyses if _is_success(a)]
        if not successful_steps:
            raise ValueError("No successful analysis scripts found. Cannot generate notebook via Codex.")

        def _sort_key(s: Dict[str, Any]):
            n = self._extract_analysis_number_from_step(s)
            return (n if n is not None else 999,)

        successful_steps = sorted(successful_steps, key=_sort_key)
        if len(successful_steps) > 1:
            successful_steps = [successful_steps[-1]]

        # Collect paths to working analysis scripts (relative to project_dir)
        script_paths: List[str] = []
        for step in successful_steps:
            path = step.get("script_path", "")
            if path and Path(path).exists():
                try:
                    rel = Path(path).relative_to(project_dir)
                    script_paths.append(str(rel))
                except ValueError:
                    script_paths.append(path)
            else:
                analysis_num = self._extract_analysis_number_from_step(step)
                if analysis_num is not None:
                    script_paths.append(f"analysis_scripts/analysis_{analysis_num}/ (contains analysis_*.py and stdout.txt)")

        # Parse question
        raw_text = question.get("text", str(question)) if isinstance(question, dict) else str(question)
        try:
            parsed = await self._parse_question_robustly(raw_text)
        except Exception:
            parsed = {"question": raw_text, "rationale": "", "procedure": "", "title": "AutoInterp Analysis Report"}
        conclusion = evaluation_results.get("conclusion", "Analysis Complete")
        if isinstance(conclusion, dict):
            conclusion = str(conclusion)
        reasoning = evaluation_results.get("reasoning", "")[:3000]

        model_info = ""
        if task_config and task_config.get("model"):
            model_info = f"\n- Model: {task_config['model'].get('name', 'Unknown')}"
        if task_config and task_config.get("dataset"):
            model_info += f"\n- Dataset: {task_config.get('dataset', 'Unknown')}"

        script_paths_str = "\n".join(f"- {p}" for p in script_paths) if script_paths else "(no paths)"
        codex_sandbox = self.config.get("codex", {}).get("sandbox", "workspace-write")

        if previous_errors:
            prompt = self._build_codex_fix_prompt(
                output_path, previous_errors[-1], previous_errors[:-1]
            )
        else:
            prompt = self._build_codex_initial_prompt(
                project_dir=project_dir,
                output_path=output_path,
                script_paths_str=script_paths_str,
                data_path=data_path,
                parsed=parsed,
                raw_text=raw_text,
                conclusion=conclusion,
                reasoning=reasoning,
                model_info=model_info,
            )

        action = "fixing" if previous_errors else "generating"
        self.logger.info(f"Invoking Codex to {action} notebook...")
        result = await asyncio.to_thread(
            run_codex_exec,
            prompt=prompt,
            cwd=project_dir,
            sandbox=codex_sandbox,
            timeout=timeout,
        )

        if result.returncode != 0:
            self.logger.warning(f"Codex exited with code {result.returncode}: {result.stderr[:500] if result.stderr else result.stdout[:500]}")
            raise RuntimeError(f"Codex failed: {result.stderr or result.stdout or 'unknown error'}")

        if not output_path.exists():
            raise FileNotFoundError(f"Codex did not create the notebook at {output_path}")

        return output_path

    def _validate_notebook_runs(
        self, nb_path: Path, timeout: int = 300
    ) -> tuple[bool, Optional[str]]:
        """
        Execute the notebook to validate it runs.

        Returns:
            (success, error_message): True and None if execution succeeds;
            False and error text (for feedback) if it fails.
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
                err = (result.stderr or result.stdout or "Unknown error").strip()
                self.logger.warning(f"Notebook validation failed: {err[:500]}")
                return False, err
            return True, None
        except subprocess.TimeoutExpired:
            self.logger.warning("Notebook validation timed out")
            return False, "Notebook execution timed out. Consider reducing computation or splitting long-running cells."
        except (FileNotFoundError, ModuleNotFoundError):
            self.logger.debug("nbconvert not found, skipping notebook validation")
            return True, None  # Don't fail if nbconvert not installed
        except Exception as e:
            self.logger.warning(f"Notebook validation error: {e}")
            return False, str(e)

    def _extract_analysis_number_from_step(self, step: Dict[str, Any]) -> Optional[int]:
        """Extract analysis number from step (e.g. 3 from script_path .../analysis_3/...)."""
        script_path = step.get("script_path", "")
        if script_path:
            match = re.search(r"analysis_(\d+)", str(script_path))
            if match:
                return int(match.group(1))
        return None

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
        Resolve the final script from the analysis_scripts directory.
        Supports both: analysis_{n}/analysis_*.py (Codex flat) and
        analysis_{n}/attempt_{m}/analysis_*.py (sequential, highest attempt).
        """
        try:
            analysis_scripts_dir = self.path_resolver.get_path("analysis_scripts")
            analysis_dir = analysis_scripts_dir / f"analysis_{analysis_num}"
            if not analysis_dir.exists():
                return ""
            script_search_dir = None
            attempt_dirs = []
            for item in analysis_dir.iterdir():
                if item.is_dir() and item.name.startswith("attempt_"):
                    try:
                        attempt_num = int(item.name.split("_")[1])
                        attempt_dirs.append((attempt_num, item))
                    except (ValueError, IndexError):
                        continue
            if attempt_dirs:
                attempt_dirs.sort(key=lambda x: x[0])
                script_search_dir = attempt_dirs[-1][1]
            else:
                script_search_dir = analysis_dir  # Flat structure (Codex)
            for f in script_search_dir.iterdir():
                if f.is_file() and f.name.startswith("analysis_") and f.suffix == ".py":
                    with open(f, "r", encoding="utf-8") as fp:
                        return fp.read()
        except Exception as e:
            self.logger.warning(f"Could not resolve final script from disk for analysis_{analysis_num}: {e}")
        return ""

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
