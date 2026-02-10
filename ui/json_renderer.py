# ui/json_renderer.py
import re
from typing import Dict, Any, List


class JSONRenderer:
    """
    Converts structured JSON responses to markdown with clickable source citations.
    """
    
    def __init__(self, json_response: Dict[str, Any], sources: Dict[str, Dict], confidence: Dict[str, Any]):
        self.json_response = json_response
        self.sources = sources
        self.confidence = confidence
        self.citation_count = 0
    
    def _replace_source_tags(self, text: str) -> str:
        """
        Replace [SOURCE_X] tags with clickable superscript citations.
        Example: "text [SOURCE_1]" -> "text <sup>[1]</sup>"
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
                pmid = source.get('pmid', '')
                
                # Create tooltip with source info
                tooltip = f"{title} ({year})"
                if pmid and pmid != "Unknown":
                    tooltip += f" | PMID: {pmid}"
                
                # Return markdown with tooltip (Streamlit will render this)
                return f'<sup><span title="{tooltip}">[{source_num}]</span></sup>'
            else:
                return f'<sup>[{source_num}]</sup>'
        
        result = re.sub(pattern, replacement, text)
        return result
    
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
        lines = [f"**{header}**"]
        
        # Check if this section has items (bullet points)
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    # Different item types
                    if "name" in item:  # Types section
                        name = item.get("name", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        details = self._replace_source_tags(item.get("details", ""))
                        
                        lines.append(f"* **{name}**: {desc}")
                        if details:
                            lines.append(f"  {details}")
                    
                    elif "symptom" in item:  # Symptoms section
                        symptom = item.get("symptom", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"* **{symptom}**: {desc}")
                    
                    elif "complication" in item:  # Complications
                        comp = item.get("complication", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        freq = item.get("frequency", "")
                        
                        if freq:
                            lines.append(f"* **{comp}** ({freq}): {desc}")
                        else:
                            lines.append(f"* **{comp}**: {desc}")
        
        # Check if this section has content (paragraph)
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"\n{content}")
        
        return "\n".join(lines) + "\n"
    
    def _render_section_complications(self, section: Dict) -> str:
        """Render sections for complications-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    comp = item.get("complication", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    freq = item.get("frequency", "")
                    
                    if freq:
                        lines.append(f"* **{comp}** ({freq}): {desc}")
                    else:
                        lines.append(f"* **{comp}**: {desc}")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"\n{content}")
        
        return "\n".join(lines) + "\n"
    
    def _render_section_comparison(self, section: Dict) -> str:
        """Render sections for comparison-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**"]
        
        # Check for comparison table
        if "comparison_table" in section:
            table = section["comparison_table"]
            
            if table and len(table) > 0:
                # Get option names from first row
                first_row = table[0]
                option_a_name = "Option A"
                option_b_name = "Option B"
                
                # Try to infer names from the comparison
                # (You could enhance this by passing option names explicitly)
                
                # Create markdown table header
                lines.append("")
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
            lines.append(f"\n{content}")
        
        return "\n".join(lines) + "\n"
    
    def _render_section_treatment(self, section: Dict) -> str:
        """Render sections for treatment-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    treatment = item.get("treatment", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    lines.append(f"* **{treatment}**: {desc}")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"\n{content}")
        
        return "\n".join(lines) + "\n"
    
    def _render_section_diagnosis(self, section: Dict) -> str:
        """Render sections for diagnosis-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    procedure = item.get("procedure", "")
                    desc = self._replace_source_tags(item.get("description", ""))
                    accuracy = item.get("accuracy", "")
                    
                    lines.append(f"* **{procedure}**: {desc}")
                    if accuracy:
                        lines.append(f"  *Accuracy: {accuracy}*")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"\n{content}")
        
        return "\n".join(lines) + "\n"
    
    def _render_section_timing(self, section: Dict) -> str:
        """Render sections for timing-type questions."""
        header = section.get("header", "")
        lines = [f"**{header}**"]
        
        if "items" in section:
            for item in section["items"]:
                if isinstance(item, dict):
                    indication = item.get("indication", "")
                    explanation = self._replace_source_tags(item.get("explanation", ""))
                    
                    # Could be "indication" or "consideration"
                    if indication:
                        lines.append(f"* **{indication}**: {explanation}")
                    elif "consideration" in item:
                        consideration = item.get("consideration", "")
                        desc = self._replace_source_tags(item.get("description", ""))
                        lines.append(f"* **{consideration}**: {desc}")
        
        elif "content" in section:
            content = self._replace_source_tags(section.get("content", ""))
            lines.append(f"\n{content}")
        
        return "\n".join(lines) + "\n"
    
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
        Render the sources section with expandable details.
        """
        lines = ["**Key Sources:**\n"]
        
        # Sort sources by number
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: int(x[0].replace("SOURCE_", ""))
        )
        
        for source_key, source in sorted_sources:
            source_num = source_key.replace("SOURCE_", "")
            title = source.get("title", "Unknown")
            year = source.get("year", "")
            pmid = source.get("pmid", "")
            evidence_level = source.get("evidence_level", "")
            
            # Format source line
            source_line = f"**[{source_num}]** {title}"
            
            if year and year != "Unknown":
                source_line += f" ({year})"
            
            details = []
            if pmid and pmid != "Unknown":
                details.append(f"PMID: {pmid}")
            if evidence_level and evidence_level != "Unknown":
                details.append(f"Evidence Level: {evidence_level}")
            
            if details:
                source_line += f" | {' | '.join(details)}"
            
            lines.append(f"* {source_line}")
        
        return "\n".join(lines)
    
    def render_confidence(self) -> str:
        """Render confidence assessment."""
        label = self.confidence.get("label", "Unknown")
        score = self.confidence.get("score", 0)
        breakdown = self.confidence.get("breakdown", "")
        
        return f"""---

**Evidence Quality:** {label} confidence ({score}/100)

*Based on: {breakdown}*"""
