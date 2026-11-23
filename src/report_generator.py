"""
MuleNetra AI - Report Generator & Export Module
Generates professional forensic investigation PDF dossiers via ReportLab,
alongside CSV data exports for compliance records.
"""

import sys
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)

def _clean_markdown_to_reportlab(text: str) -> str:
    """Safely convert markdown bold/italic into ReportLab XML tags and escape ampersands."""
    text = text.replace("&", "&amp;")
    # Convert **bold** to <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Convert *italic* to <i>italic</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import REPORTS_DIR, APP_TITLE

class ReportGenerator:
    """
    Forensic AML Dossier and Compliance Audit Report Generator.
    """
    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_account_pdf_report(
        self,
        account_id: str,
        account_profile: Dict[str, Any],
        brief: Dict[str, str],
        transactions: pd.DataFrame
    ) -> Path:
        """
        Builds a comprehensive forensic PDF investigation research dossier.
        """
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_filename = f"MuleNetra_Forensic_Report_{account_id}_{timestamp_str}.pdf"
        output_path = self.reports_dir / report_filename

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )

        styles = getSampleStyleSheet()
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A")
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748B")
        )
        h2_style = ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "BodyCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#334155")
        )
        alert_style = ParagraphStyle(
            "AlertText",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#BE123C")
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("<b>MULENETRA AI</b> — FINANCIAL INTELLIGENCE UNIT", subtitle_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph("FORENSIC INVESTIGATION DOSSIER", title_style))
        story.append(Paragraph(f"CONFIDENTIAL COMPLIANCE AUDIT • CASE REF: FIU-{account_id}-{datetime.utcnow().strftime('%Y%m')}", subtitle_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

        # 2. Executive Metadata Box
        risk_tier = account_profile.get("risk_tier", "LOW")
        risk_score = account_profile.get("risk_score", 0.0)
        mule_role = account_profile.get("mule_role", "BENIGN")

        meta_data = [
            [
                Paragraph("<b>TARGET ACCOUNT</b>", subtitle_style),
                Paragraph(f"<b>{account_id}</b>", body_style),
                Paragraph("<b>INVESTIGATION STATUS</b>", subtitle_style),
                Paragraph("<font color='#BE123C'><b>UNDER REVIEW (SAR READY)</b></font>", body_style)
            ],
            [
                Paragraph("<b>HYBRID RISK INDEX</b>", subtitle_style),
                Paragraph(f"<b>{risk_score:.2f} / 1.00 ({risk_tier})</b>", body_style),
                Paragraph("<b>GRAPH TOPOLOGY ROLE</b>", subtitle_style),
                Paragraph(f"<b>{mule_role}</b>", body_style)
            ],
            [
                Paragraph("<b>CUMULATIVE INFLOW</b>", subtitle_style),
                Paragraph(f"${account_profile.get('total_inflow', 0):,.2f}", body_style),
                Paragraph("<b>CUMULATIVE OUTFLOW</b>", subtitle_style),
                Paragraph(f"${account_profile.get('total_outflow', 0):,.2f}", body_style)
            ],
            [
                Paragraph("<b>COUNTERPARTIES (IN/OUT)</b>", subtitle_style),
                Paragraph(f"{account_profile.get('in_degree', 0)} in / {account_profile.get('out_degree', 0)} out", body_style),
                Paragraph("<b>DOSSIER TIMESTAMP</b>", subtitle_style),
                Paragraph(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), body_style)
            ]
        ]

        t_meta = Table(meta_data, colWidths=[130, 135, 130, 135])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 14))

        # 3. Executive Incident Summary
        story.append(Paragraph("1. Executive Incident Summary", h2_style))
        story.append(Paragraph(_clean_markdown_to_reportlab(brief.get("executive_summary", "No summary available.")), body_style))
        story.append(Spacer(1, 10))

        # 4. Factual Forensic Observations
        story.append(Paragraph("2. Factual Forensic Observations", h2_style))
        for line in brief.get("factual_observations", "").split("\n"):
            if line.strip():
                clean_line = _clean_markdown_to_reportlab(line)
                story.append(Paragraph(clean_line, body_style))
                story.append(Spacer(1, 2))
        story.append(Spacer(1, 8))

        # 5. Investigative Hypotheses & Topology
        story.append(Paragraph("3. Investigative Hypotheses & Mule Typology", h2_style))
        for line in brief.get("hypotheses", "").split("\n"):
            if line.strip():
                clean_line = _clean_markdown_to_reportlab(line)
                story.append(Paragraph(clean_line, body_style))
                story.append(Spacer(1, 2))
        story.append(Spacer(1, 8))

        # 6. Associated Transactions Snapshot
        story.append(Paragraph("4. Material Transaction Records (Audited Sample)", h2_style))
        if len(transactions) > 0:
            tx_sample = transactions.head(8)
            tx_table_data = [["TX ID", "Date", "Sender", "Receiver", "Amount ($)", "Type", "Risk Score"]]
            for _, r in tx_sample.iterrows():
                tx_table_data.append([
                    str(r.get("tx_id", "")),
                    str(r.get("timestamp", ""))[:16],
                    str(r.get("sender_account", "")),
                    str(r.get("receiver_account", "")),
                    f"{float(r.get('amount', 0)):,.2f}",
                    str(r.get("tx_type", "")),
                    f"{float(r.get('hybrid_risk_score', 0)):.2f}"
                ])
            
            t_tx = Table(tx_table_data, colWidths=[70, 85, 75, 75, 75, 65, 65])
            t_tx.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
                ('ALIGN', (4,0), (4,-1), 'RIGHT'),
                ('ALIGN', (6,0), (6,-1), 'CENTER'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(t_tx)
        else:
            story.append(Paragraph("No direct transactions recorded.", body_style))
        story.append(Spacer(1, 10))

        # 7. Recommended Next Actions
        story.append(Paragraph("5. Recommended Compliance Actions", h2_style))
        for line in brief.get("recommended_actions", "").split("\n"):
            if line.strip():
                clean_line = _clean_markdown_to_reportlab(line)
                story.append(Paragraph(clean_line, body_style))
                story.append(Spacer(1, 2))
        story.append(Spacer(1, 14))

        # 8. Forensic Certification & Sign-off Block
        sign_data = [
            [
                Paragraph("<b>LEAD INVESTIGATOR SIGN-OFF</b><br/><br/>___________________________<br/>Compliance Intelligence Officer", subtitle_style),
                Paragraph("<b>SUPERVISORY REVIEW</b><br/><br/>___________________________<br/>Chief Anti-Money Laundering Officer", subtitle_style)
            ]
        ]
        t_sign = Table(sign_data, colWidths=[265, 265])
        t_sign.setStyle(TableStyle([
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t_sign)

        # Build PDF document
        doc.build(story)
        return output_path

    def export_data_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Export pandas DataFrame to CSV inside reports directory."""
        output_path = self.reports_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

if __name__ == "__main__":
    from src.data_loader import get_or_create_dataset
    from src.risk_engine import RiskScoringEngine
    from src.genai_assistant import GenAIAssistant
    
    df = get_or_create_dataset()
    engine = RiskScoringEngine()
    scored_tx = engine.score_transactions(df)
    profiles = engine.compute_account_risk_profiles(scored_tx)
    
    rep_gen = ReportGenerator()
    csv_path = rep_gen.export_data_to_csv(profiles, "Account_Risk_Profiles_Export.csv")
    print(f"Exported CSV to {csv_path}")
    
    top_acc = profiles.iloc[0]["account_id"]
    acc_prof = profiles.iloc[0].to_dict()
    assistant = GenAIAssistant()
    acc_tx = scored_tx[(scored_tx["sender_account"] == top_acc) | (scored_tx["receiver_account"] == top_acc)]
    brief = assistant.generate_investigation_brief(top_acc, acc_prof, acc_tx.to_dict("records"), {})
    
    pdf_path = rep_gen.generate_account_pdf_report(top_acc, acc_prof, brief, acc_tx)
    print(f"Generated PDF Report: {pdf_path}")
