# ui/json_renderer.py
import re
from typing import Dict, Any, List


class JSONRenderer:
    """
    Converts structured JSON responses to markdown with clickable source citations.
    """
    
    def __init__(self, json_response: Dict[str, Any], sources: Dict[str, Dict], confidence: Dict[str, Any], faithfulness: Dict[str, Any] = None):
        self.json_response = json_response
        self.sources = sources
        self.confidence = confidence
        self.faithfulness = faithfulness or {}
        self.citation_count = 0
    
    def _replace_source_tags(self, text: str) -> str:
        """
        Replace [SOURCE_X] tags with clickable anchor-linked citations.
        Example: "text [SOURCE_1]" -> "text <a href='#source-1'>[1]</a>"
        """
        if not text:
            return ""
        
        # Find all [SOURCE_X] tags
        pattern = r'\[SOURCE_(\d+)\]'
        
        def replacement(match):
            source_num = match.group(1)
            source_key = f"SOURCE_{source_num}"
            
            if source_key in self.sources:
                source = self.sources[source_key]
                title = source.get('title', 'Unknown')
                year = source.get('year', '')
                
                # Create clickable citation that opens details and scrolls to source
                # Using onclick to expand the details element
                return (
                    f'<a href="#source-{source_num}" '
                    f'class="citation-link" '
                    f'onclick="document.getElementById(\'sources-section\').open=true;" '
                    f'title="{title} ({year})">'
                    f'[{source_num}]</a>'
                )
            else:
                return f'[{source_num}]'
        
        result = re.sub(pattern, replacement, text)
        return result
    
    def _clean_frequency(self, freq: str) -> str:
        """
        Remove 'Not quantified in available excerpts' and similar phrases.
        Returns empty string if that's all the frequency contained.
        """
        if not freq:
            return ""
        
        # List of phrases to remove
        unwanted_phrases = [
            "Not quantified in available excerpts",
            "Not specified",
            "Not mentioned",
            "Not quantified",
            "(Not specified)",
            "(Not quantified)",
        ]
        
        cleaned = freq
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        # Clean up whitespace
        cleaned = cleaned.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # If nothing left or just parentheses, return empty
        if not cleaned or cleaned in ['()', '( )']:
            return ""
        
        return cleaned
    
    def _render_faithfulness_badge(self) -> str:
        """
        Render faithfulness score badge.
        
        Returns:
            HTML/markdown badge showing faithfulness score
        """
        if not self.faithfulness or self.faithfulness.get("score") is None:
            # No faithfulness data or evaluation failed
            error = self.faithfulness.get("error", "")
            if error:
                return '<div class="faithfulness-badge unavailable">⚠️ Faithfulness: Not Available</div>'
            return ""  # Don't show anything if no data
        
        score = self.faithfulness.get("score", 0)
        label = self.faithfulness.get("label", "Unknown")
        percentage = int(score * 100)
        
        # Determine CSS class and icon based on score
        # Updated thresholds: 80%+ = High, 60-79% = Medium, <60% = Low
        if score >= 0.80:  # Changed from 0.90
            css_class = "high"
            icon = "✅"
        elif score >= 0.60:  # Changed from 0.70
            css_class = "medium"
            icon = "⚠️"
        else:
            css_class = "low"
            icon = "❌"
        
        # Build badge with tooltip
        total = self.faithfulness.get("total_statements", 0)
        evaluated = self.faithfulness.get("evaluated_statements", 0)
        tooltip = f"{evaluated}/{total} statements verified"
        
        return f'''<div class="faithfulness-badge {css_class}" title="{tooltip}">
{icon} <strong>Faithfulness:</strong> {percentage}% ({label})
</div>'''
    
    def _render_faithfulness_for_sources(self) -> str:
        """
        Render faithfulness as a quality box in sources section (similar to evidence quality).
        
        Returns:
            HTML div showing faithfulness score, styled like evidence quality box
        """
        if not self.faithfulness or self.faithfulness.get("score") is None:
            # No faithfulness data or evaluation failed
            error = self.faithfulness.get("error", "")
            if error:
                return '''
<div class="faithfulness-quality">
<strong>Answer Faithfulness:</strong> Not Available
<br>
<em>Evaluation could not be completed</em>
</div>'''
            return ""  # Don't show anything if no data
        
        score = self.faithfulness.get("score", 0)
        label = self.faithfulness.get("label", "Unknown")
        percentage = int(score * 100)
        
        # Get details for description text
        total = self.faithfulness.get("total_statements", 0)
        evaluated = self.faithfulness.get("evaluated_statements", 0)
        
        # Determine color scheme based on score
        if score >= 0.80:
            color_class = "high"  # Green
        elif score >= 0.60:
            color_class = "medium"  # Yellow
        else:
            color_class = "low"  # Red
        
        # Build HTML div styled like evidence quality box
        return f'''
<div class="faithfulness-quality {color_class}">
<strong>Answer Faithfulness:</strong> {percentage}% ({label})
<br>
<em>Based on: {evaluated}/{total} statements verified against sources</em>
</div>'''
    
    def _render_overview(self) -> str:
        """Render the overview section."""
        overview = self.json_response.get("overview", "")
        overview_with_citations = self._replace_source_tags(overview)
        
        return f"""**AI Overview**

{overview_with_citations}

---
"""
    
    def _render_section_definition(self, section: Dict) -> str:
        """Render sections for definition-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**\n"]
        
        # Check if this section has items (bullet points)
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    # Different item types
                    if "name" in item:  # Types section
                        name = item.get("name", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        details = self._replace_source_tags(item.get("details", ""))
                        
                        lines.append(f"**{name}**: {desc}")
                        if details:
                            lines.append(f"{details}")
                        lines.append("")  # Spacing
                    
                    elif "symptom" in item:  # Symptoms section
                        symptom = item.get("symptom", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"**{symptom}**: {desc}")
                        lines.append("")
                    
                    elif "complication" in item:  # Complications
                        comp = item.get("complication", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        freq = self._clean_frequency(item.get("frequency", ""))
                        
                        if freq:
                            lines.append(f"**{comp}** ({freq}): {desc}")
                        else:
                            lines.append(f"**{comp}**: {desc}")
                        lines.append("")
        
        # Check if this section has content (paragraph)
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_section_complications(self, section: Dict) -> str:
        """Render sections for complications-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**\n"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    comp = item.get("complication", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    freq = self._clean_frequency(item.get("frequency", ""))
                    
                    if freq:
                        lines.append(f"**{comp}** ({freq}): {desc}")
                    else:
                        lines.append(f"**{comp}**: {desc}")
                    lines.append("")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_section_comparison(self, section: Dict) -> str:
        """Render sections for comparison-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**\n"]
        
        # Check for comparison table
        if "comparison_table" in section:
            table = section["comparison_table"]
            
            if table and len(table) > 0:
                # Create markdown table header
                lines.append("| Aspect | First Option | Second Option |")
                lines.append("|--------|-------------|---------------|")
                
                for row in table:
                    aspect = row.get("aspect", "")
                    opt_a = self._replace_source_tags(row.get("option_a", ""))
                    opt_b = self._replace_source_tags(row.get("option_b", ""))
                    
                    lines.append(f"| **{aspect}** | {opt_a} | {opt_b} |")
                
                lines.append("")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_section_treatment(self, section: Dict) -> str:
        """Render sections for treatment-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**\n"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    treatment = item.get("treatment", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    lines.append(f"**{treatment}**: {desc}")
                    lines.append("")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_section_diagnosis(self, section: Dict) -> str:
        """Render sections for diagnosis-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**\n"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    procedure = item.get("procedure", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    accuracy = item.get("accuracy", "")
                    
                    lines.append(f"**{procedure}**: {desc}")
                    if accuracy:
                        lines.append(f"*Accuracy: {accuracy}*")
                    lines.append("")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_section_timing(self, section: Dict) -> str:
        """Render sections for timing-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**\n"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    indication = item.get("indication", "")
                    explanation = self._replace_source_tags(item.get("explanation", ""))
                    
                    # Could be "indication" or "consideration"
                    if indication:
                        lines.append(f"**{indication}**: {explanation}")
                        lines.append("")
                    elif "consideration" in item:
                        consideration = item.get("consideration", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"**{consideration}**: {desc}")
                        lines.append("")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"{content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def render(self, question_type: str) -> str:
        """
        Render the complete response based on question type.
        """
        markdown_parts = []
        
        # Render overview
        markdown_parts.append(self._render_overview())
        
        # Render sections based on question type
        sections = self.json_response.get("sections", [])
        
        for section in sections:
            if question_type == "definition":
                markdown_parts.append(self._render_section_definition(section))
            elif question_type == "complications":
                markdown_parts.append(self._render_section_complications(section))
            elif question_type == "comparison":
                markdown_parts.append(self._render_section_comparison(section))
            elif question_type == "treatment":
                markdown_parts.append(self._render_section_treatment(section))
            elif question_type == "diagnosis":
                markdown_parts.append(self._render_section_diagnosis(section))
            elif question_type == "timing":
                markdown_parts.append(self._render_section_timing(section))
            else:
                # Fallback
                markdown_parts.append(self._render_section_definition(section))
        
        return "\n".join(markdown_parts)
    
    def render_sources(self) -> str:
        """
        Render the sources section as collapsible HTML details element.
        """
        # Sort sources by number
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: int(x[0].replace("SOURCE_", ""))
        )
        
        # Build source list with anchor IDs
        source_lines = []
        for source_key, source in sorted_sources:
            source_num = source_key.replace("SOURCE_", "")
            title = source.get("title", "Unknown")
            year = source.get("year", "")
            pmid = source.get("pmid", "")
            evidence_level = source.get("evidence_level", "")
            
            # Format source line with anchor ID for scrolling
            source_line = f'<div id="source-{source_num}" class="source-item">'
            source_line += f"<strong>[{source_num}]</strong> {title}"
            
            if year and year != "Unknown":
                source_line += f" ({year})"
            
            details = []
            if pmid and pmid != "Unknown":
                # Make PMID clickable link to PubMed
                pmid_link = f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">PMID: {pmid}</a>'
                details.append(pmid_link)
            if evidence_level and evidence_level != "Unknown":
                details.append(f"Evidence Level: {evidence_level}")
            
            if details:
                source_line += f" | {' | '.join(details)}"
            
            source_line += "</div>"
            source_lines.append(source_line)
        
        # Get confidence info
        label = self.confidence.get("label", "Unknown")
        score = self.confidence.get("score", 0)
        breakdown = self.confidence.get("breakdown", "")
        
        # Get faithfulness info if available
        faithfulness_html = self._render_faithfulness_for_sources()
        
        # Build collapsible HTML details element
        html = f"""
<details id="sources-section" class="sources-collapsible">
<summary class="sources-summary">📚 Show Sources</summary>

<div class="sources-content">

<div class="evidence-quality">
<strong>Evidence Quality:</strong> {label} confidence ({score}/100)
<br>
<em>Based on: {breakdown}</em>
</div>

{faithfulness_html}

<div class="sources-divider"></div>

<div class="sources-list">
<strong>Key Sources:</strong>
<br><br>
{'<br>'.join(source_lines)}
</div>

</div>
</details>
"""
        
        return html
    
    def render_confidence(self) -> str:
        """
        This is now included in render_sources(), so return empty.
        """
        return ""
