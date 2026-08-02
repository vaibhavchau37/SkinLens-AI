import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_medical_pdf(output_pdf_path, data):
    """
    Generates a professional hospital-style screening report in PDF format.
    
    Args:
        output_pdf_path (str or Path): Target path to write the output PDF.
        data (dict): Dictionary containing patient details, scan metrics, and paths.
    """
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom hospital styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor('#0284c7'), # Professional medical blue
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    
    label_style = ParagraphStyle(
        'GridLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#334155')
    )
    
    value_style = ParagraphStyle(
        'GridValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#0f172a')
    )
    
    rec_title_style = ParagraphStyle(
        'RecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=4
    )
    
    rec_text_style = ParagraphStyle(
        'RecText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.HexColor('#334155'),
        leading=13
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        textColor=colors.HexColor('#64748b'),
        leading=11
    )
    
    story = []
    
    # 1. Hospital-style Header Block
    story.append(Paragraph("SKINLENS AI CLINICAL DIAGNOSTIC SCREENING REPORT", title_style))
    story.append(Paragraph("Automated Deep Neural Network Classification Evidence Sheet", subtitle_style))
    
    # 2. Structured Metadata Grid Table
    info_data = [
        [
            Paragraph("Patient Name:", label_style), Paragraph(data.get("patient_name", "Anonymous Subject"), value_style),
            Paragraph("Patient ID / ID Record:", label_style), Paragraph(data.get("stem", "N/A").upper()[:12], value_style)
        ],
        [
            Paragraph("Age / Gender:", label_style), Paragraph(f"{data.get('patient_age', 'N/A')} yrs / {data.get('patient_gender', 'Unspecified')}", value_style),
            Paragraph("Date Processed:", label_style), Paragraph(data.get("timestamp", "N/A"), value_style)
        ],
        [
            Paragraph("Responsible Unit:", label_style), Paragraph(data.get("clinician_name", "Dermatology Department"), value_style),
            Paragraph("Model Architecture:", label_style), Paragraph(data.get("model_name", "EfficientNet-B0 Baseline"), value_style)
        ],
        [
            Paragraph("Inference Device:", label_style), Paragraph(data.get("device", "CPU"), value_style),
            Paragraph("Processing Speed:", label_style), Paragraph(f"{data.get('execution_time_ms', 0.0):.1f} ms", value_style)
        ]
    ]
    
    info_table = Table(info_data, colWidths=[90, 180, 110, 160])
    info_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#f8fafc')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))
    
    # 3. Highlighted AI Classification Card
    label = data.get("label", "Benign")
    prob_pct = f"{data.get('probability', 0.0):.2%}"
    conf_pct = f"{data.get('confidence', 0.0):.2%}"
    
    bg_color = colors.HexColor('#fff1f2') if label == "Malignant" else colors.HexColor('#ecfdf5')
    text_color = colors.HexColor('#be123c') if label == "Malignant" else colors.HexColor('#047857')
    border_color = colors.HexColor('#fda4af') if label == "Malignant" else colors.HexColor('#a7f3d0')
    
    card_title_style = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#475569')
    )
    
    card_val_style = ParagraphStyle(
        'CardVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=text_color
    )
    
    metrics_style = ParagraphStyle(
        'Metrics',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#1e293b')
    )
    
    card_data = [
        [
            Paragraph("AI Classification output:", card_title_style),
            Paragraph(f"{label.upper()} (Malignancy Probability: {prob_pct})", card_val_style)
        ],
        [
            Paragraph("Decision Confidence:", card_title_style),
            Paragraph(f"{conf_pct} (Absolute distance to decision boundary)", metrics_style)
        ]
    ]
    
    card_table = Table(card_data, colWidths=[150, 390])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1.5, border_color),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 15))
    
    # 4. Clinician Observations & Recommendation Block
    recommendation = data.get("recommendation", "")
    if not recommendation:
        if label == "Malignant":
            recommendation = "Biopsy & Referral Recommended: This lesion has structural features suggesting malignancy. Urgent clinical evaluation by a dermatologist is recommended."
        else:
            recommendation = "Periodic Self-Screening: The lesion is classified as benign. Patients should continue standard monitoring (ABCDE rule) and report modifications to a doctor."
            
    notes = data.get("notes", "")
    combined_rec_notes = recommendation
    if notes:
        combined_rec_notes = f"{combined_rec_notes}<br/><br/><b>Clinician Diagnostic Observations:</b> {notes}"
        
    rec_data = [
        [
            Paragraph("CLINICAL RECOMMENDATION & OBSERVATIONS", rec_title_style)
        ],
        [
            Paragraph(combined_rec_notes, rec_text_style)
        ]
    ]
    rec_table = Table(rec_data, colWidths=[540])
    rec_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 15))
    
    # 5. Visual Proof Section (Original and Grad-CAM side-by-side)
    img_w = 230
    img_h = 230
    
    img_path = data.get("image_path")
    overlay_path = data.get("overlay_path")
    
    has_img = img_path and os.path.exists(img_path)
    has_overlay = overlay_path and os.path.exists(overlay_path)
    
    row_images = []
    row_labels = []
    
    if has_img:
        row_images.append(Image(str(img_path), width=img_w, height=img_h))
        row_labels.append(Paragraph("ORIGINAL LESION IMAGE", label_style))
    else:
        row_images.append(Paragraph("No original image uploaded.", value_style))
        row_labels.append(Paragraph("ORIGINAL LESION IMAGE", label_style))
        
    if has_overlay:
        row_images.append(Image(str(overlay_path), width=img_w, height=img_h))
        row_labels.append(Paragraph("GRAD-CAM ATTENTION OVERLAY", label_style))
    else:
        row_images.append(Paragraph("No Grad-CAM overlay generated.", value_style))
        row_labels.append(Paragraph("GRAD-CAM ATTENTION OVERLAY", label_style))
        
    img_table_data = [
        row_labels,
        row_images
    ]
    
    img_table = Table(img_table_data, colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))
    
    # 6. Medical Disclaimer
    disclaimer_text = (
        "<b>Clinical Notice & Disclaimer:</b> This document displays analytical output generated by artificial intelligence. "
        "Software predictions do not establish clinical fact. Final medical diagnoses and treatment plans must be correlated "
        "with clinical manifestations, patient history, dermoscopy, and tissue biopsy assessments by qualified dermatology "
        "professionals. SkinLens AI is a clinical decision support tool and does not guarantee the absence or presence of malignancy."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 20))
    
    # 7. Signature Block
    sig_data = [
        [
            Paragraph(f"Model ID: {data.get('model_name', 'EfficientNet-B0')}", value_style),
            Paragraph("Clinician Signature: _______________________", value_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(sig_table)
    
    # Build the document
    doc.build(story)
