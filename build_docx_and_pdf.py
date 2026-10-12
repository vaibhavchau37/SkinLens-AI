import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_docx():
    doc = docx.Document()

    # Set page margins (0.6 inch)
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    def set_cell_background(cell, fill_hex):
        tcPr = cell._element.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def add_table_header(table, headers, col_widths=None):
        hdr_cells = table.rows[0].cells
        for i, header_text in enumerate(headers):
            hdr_cells[i].text = header_text
            set_cell_background(hdr_cells[i], "0284C7")
            p = hdr_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9.5)

    # Title Header
    p_hdr = doc.add_paragraph()
    p_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_u = p_hdr.add_run("P P SAVANI UNIVERSITY - SCHOOL OF ENGINEERING\n")
    run_u.bold = True
    run_u.font.size = Pt(13)
    run_u.font.color.rgb = RGBColor(16, 44, 87)

    run_t = p_hdr.add_run("INTERNSHIP / UDP / TRAINING REPORT – REPORTING 4\n")
    run_t.bold = True
    run_t.font.size = Pt(15)
    run_t.font.color.rgb = RGBColor(2, 132, 199)

    run_s = p_hdr.add_run("(Final Project Completion, Demonstration & Evaluation Report)")
    run_s.italic = True
    run_s.font.size = Pt(10.5)
    run_s.font.color.rgb = RGBColor(71, 85, 105)

    # 1. Student Information
    h1 = doc.add_heading("1. Student Information", level=2)
    h1.runs[0].font.color.rgb = RGBColor(2, 132, 199)
    
    t1 = doc.add_table(rows=6, cols=2)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_info = [
        ("Student Name", "Vaibhavkumar Chaudhary"),
        ("Enrollment Number", "23SE02ML030"),
        ("Program/Branch", "B.Tech , Computer Science Engineering (ML & AI)"),
        ("Semester", "7th"),
        ("Contact Number", "6355072709"),
        ("Email ID", "23se02ml030@ppsu.ac.in")
    ]
    for row_idx, (label, val) in enumerate(s_info):
        row = t1.rows[row_idx]
        row.cells[0].text = label
        row.cells[1].text = val
        set_cell_background(row.cells[0], "F1F5F9")
        row.cells[0].paragraphs[0].runs[0].bold = True

    # 2. Internship / UDP / Training Details
    h2 = doc.add_heading("2. Internship / UDP / Training Details", level=2)
    h2.runs[0].font.color.rgb = RGBColor(2, 132, 199)
    
    t2 = doc.add_table(rows=8, cols=2)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    u_info = [
        ("Type", "User-defined Project(UDP)- Placement Track"),
        ("Company/Organization", "User-defined Project,SOE, P P Savani University"),
        ("Department/Domain", "Artificial Intelligence / Machine Learning"),
        ("Mentor Name (Company)", "N/A (Self-Defined Project)"),
        ("Mentor Name (Institute)", "Mr.Kushal Singh"),
        ("Start Date", "27 June 2026"),
        ("End Date", "26 October 2026"),
        ("Project Mode", "UDP Placement Track - Final Evaluation")
    ]
    for row_idx, (label, val) in enumerate(u_info):
        row = t2.rows[row_idx]
        row.cells[0].text = label
        row.cells[1].text = val
        set_cell_background(row.cells[0], "F1F5F9")
        row.cells[0].paragraphs[0].runs[0].bold = True

    # 3. Work Progress Summary
    h3 = doc.add_heading("3. Work Progress Summary", level=2)
    h3.runs[0].font.color.rgb = RGBColor(2, 132, 199)

    doc.add_paragraph("a) Project Title:").runs[0].bold = True
    p_title = doc.add_paragraph("SkinLens AI : Intelligent Skin Lesion Analysis & Early Skin Cancer Detection")
    p_title.runs[0].bold = True
    p_title.runs[0].font.color.rgb = RGBColor(15, 23, 42)

    doc.add_paragraph("b) Brief Overview of Work Completed After Reporting 3:").runs[0].bold = True

    bullets = [
        "Major tasks completed: Completed Grad-CAM Explainable AI (XAI) feature activation visualizer; integrated multi-heuristic input quality gate (skin tone ratio + document detector); created hospital-grade PDF report generator; added Test-Time Augmentation (TTA) and confidence meter; expanded Streamlit UI dashboard.",
        "Modules completed: Grad-CAM Explainable AI Module, Pre-prediction Quality Gate & Validation Module, Clinical PDF Report Generation Module, TTA Inference Engine, Scan History & Persistence Storage Module, Model Architecture Inspector Module, Full Streamlit Clinical Dashboard Module.",
        "New features implemented: Grad-CAM heatmap & 0.65 alpha-blend overlay, pre-prediction non-skin/document rejection filter, decision-margin confidence meter, TTA multi-view inference, automated PDF screening sheet export, checkpoint manager (.pth load/reset), patient demographic metadata entry, interactive confusion matrix & model training curves.",
        "Research / development work completed: Researched Grad-CAM visual explainability on final conv layer (features[7]) of EfficientNet-B0; engineered YCrCb skin-tone thresholding (Y in [0,255], Cr in [135,180], Cb in [85,135]) and white-background document detector (>45% white ratio); evaluated multi-view TTA strategies.",
        "Integration work completed: Connected Streamlit UI -> Image Quality Gate -> Preprocessor -> PyTorch DataLoader -> EfficientNet-B0 Model -> Logit/Sigmoid Engine -> Grad-CAM Heatmap Visualizer -> ReportLab PDF Builder -> File System Persistence."
    ]
    for bullet in bullets:
        doc.add_paragraph(bullet, style='List Bullet')

    doc.add_paragraph("c) Overall Project Completion Status:").runs[0].bold = True
    t_status = doc.add_table(rows=8, cols=3)
    t_status.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Component", "Status", "Completion (%)"]
    for i, h in enumerate(headers):
        t_status.rows[0].cells[i].text = h
        set_cell_background(t_status.rows[0].cells[i], "0284C7")
        t_status.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        t_status.rows[0].cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    status_data = [
        ("Requirement / Analysis", "Completed", "100%"),
        ("Design", "Completed", "100%"),
        ("Development", "Completed", "100%"),
        ("Integration", "Completed", "100%"),
        ("Testing", "Completed", "100%"),
        ("Documentation", "Completed", "100%"),
        ("Deployment", "Completed (Prototype Web App ready for production deployment)", "100%")
    ]
    for row_idx, data in enumerate(status_data, start=1):
        for col_idx, val in enumerate(data):
            t_status.rows[row_idx].cells[col_idx].text = val

    p_tot = doc.add_paragraph("Overall Project Completion: 100 %")
    p_tot.runs[0].bold = True
    p_tot.runs[0].font.color.rgb = RGBColor(4, 120, 87)

    # 4. Advanced Implementation
    h4 = doc.add_heading("4. Advanced Implementation", level=2)
    h4.runs[0].font.color.rgb = RGBColor(2, 132, 199)
    doc.add_paragraph("a) Module / Feature Implementation:").runs[0].bold = True

    t_mod = doc.add_table(rows=6, cols=3)
    t_mod.alignment = WD_TABLE_ALIGNMENT.CENTER
    mod_headers = ["Module / Feature", "Work Completed", "Status"]
    for i, h in enumerate(mod_headers):
        t_mod.rows[0].cells[i].text = h
        set_cell_background(t_mod.rows[0].cells[i], "0284C7")
        t_mod.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        t_mod.rows[0].cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    mod_data = [
        ("Grad-CAM Explainable AI Module", "Implemented PyTorch forward/backward hooks on final conv layer of EfficientNet-B0 (features[7]) to extract activation maps and gradients. Generated JET colormap heatmaps and 0.65 alpha-blended overlays.", "Completed"),
        ("Input Quality Gate & Skin Validation Filter", "Implemented OpenCV YCrCb skin color space thresholding (skin fraction >= 3%) and white document detector (white ratio > 45%) to halt inference on invalid images.", "Completed"),
        ("Clinical Diagnostic PDF Report Generator", "Developed automated PDF generator using ReportLab (pdf_generator.py) producing professional clinical screening sheets with patient metadata, AI risk metrics, dual image proof, and signature blocks.", "Completed"),
        ("Test-Time Augmentation (TTA) & Confidence Meter", "Implemented multi-view inference (predict_with_tta) averaging prediction probabilities across augmented image variations. Calculated decision-margin confidence as distance from decision boundary.", "Completed"),
        ("Interactive Streamlit Clinical Dashboard", "Developed full multi-page Streamlit web app (app.py) featuring Dashboard metrics, Prediction interface with real-time XAI overlay toggle, Scan History table, Model Inspector, and Documentation tabs.", "Completed")
    ]
    for row_idx, data in enumerate(mod_data, start=1):
        for col_idx, val in enumerate(data):
            t_mod.rows[row_idx].cells[col_idx].text = val

    doc.add_paragraph("b) Integration / Development Details:").runs[0].bold = True
    int_bullets = [
        "Module integration: Image Quality Gate verifies input before model execution; EfficientNet-B0 forward pass and Grad-CAM hooks execute concurrently; prediction metrics feed into ReportLab PDF generator and file persistence log.",
        "API integration: Built modular Python functions in predict.py, gradcam.py, and pdf_generator.py ready for REST API exposure via FastAPI / Flask.",
        "Database integration: File-system based JSON/TXT metadata storage and report logging in outputs/ directory. Scan history persists across sessions.",
        "Hardware integration: Automatic CUDA GPU acceleration when available (torch.cuda.is_available()), with seamless CPU fallback.",
        "Software integration: Python 3.10+, PyTorch 2.x, TorchVision, OpenCV, Albumentations, ReportLab, Streamlit, NumPy, Pandas, Matplotlib, SciPy, PIL.",
        "Major coding work: Developed gradcam.py, pdf_generator.py, updated predict.py, and built full app.py (~1580 lines) UI flow."
    ]
    for bullet in int_bullets:
        doc.add_paragraph(bullet, style='List Bullet')

    doc.add_paragraph("c) Final / Updated Architecture:").runs[0].bold = True
    doc.add_paragraph("Workflow Description:\nStreamlit UI -> Input Quality Gate (OpenCV Skin/Document Check) -> Albumentations Preprocessing (224x224, ImageNet Normalization) -> PyTorch EfficientNet-B0 Model -> Logit/Sigmoid Malignancy Score -> Grad-CAM XAI Heatmap Generator -> Decision & Confidence Metrics -> ReportLab PDF Screening Sheet Export -> Outputs File Persistence.")

    # 5. Final Testing and Validation
    h5 = doc.add_heading("5. Final Testing and Validation", level=2)
    h5.runs[0].font.color.rgb = RGBColor(2, 132, 199)

    t_test = doc.add_table(rows=6, cols=5)
    t_test.alignment = WD_TABLE_ALIGNMENT.CENTER
    test_headers = ["Testing Area", "Tests Conducted", "Passed", "Failed", "Final Status"]
    for i, h in enumerate(test_headers):
        t_test.rows[0].cells[i].text = h
        set_cell_background(t_test.rows[0].cells[i], "0284C7")
        t_test.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        t_test.rows[0].cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    test_data = [
        ("Functional Testing", "15", "15", "0", "Passed"),
        ("Integration Testing", "10", "10", "0", "Passed"),
        ("System Testing", "12", "12", "0", "Passed"),
        ("Performance Testing", "8", "8", "0", "Passed (Avg Latency: ~950ms CPU, ~85ms GPU)"),
        ("User Acceptance Testing", "10", "10", "0", "Passed")
    ]
    for row_idx, data in enumerate(test_data, start=1):
        for col_idx, val in enumerate(data):
            t_test.rows[row_idx].cells[col_idx].text = val

    # 6. Results and Performance Analysis
    h6 = doc.add_heading("6. Results and Performance Analysis", level=2)
    h6.runs[0].font.color.rgb = RGBColor(2, 132, 199)

    doc.add_paragraph("Critical Bugs / Issues:").runs[0].bold = True
    t_bugs = doc.add_table(rows=5, cols=4)
    t_bugs.alignment = WD_TABLE_ALIGNMENT.CENTER
    bug_headers = ["Issue", "Severity", "Resolution", "Final Status"]
    for i, h in enumerate(bug_headers):
        t_bugs.rows[0].cells[i].text = h
        set_cell_background(t_bugs.rows[0].cells[i], "0284C7")
        t_bugs.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        t_bugs.rows[0].cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    bug_data = [
        ("High training accuracy vs initial ROC-AUC imbalance", "High", "Implemented Focal Loss (alpha=0.25, gamma=2.0), positive class weighting, patient-wise split, and Youden's Index threshold tuning.", "Resolved"),
        ("Memory leak during repeated Grad-CAM hook calls", "Medium", "Updated gradcam.py to explicitly remove forward and backward hooks immediately post-inference.", "Resolved"),
        ("False positive predictions on document scans / screenshots", "High", "Added multi-heuristic quality gate (skin tone fraction >= 3% and document white ratio > 45%) to halt non-skin inference.", "Resolved"),
        ("Checkpoint state_dict key mismatch on custom model load", "Medium", "Refactored predict.py with key cleaning (module. prefix stripping) and strict weight matrix shape validation.", "Resolved")
    ]
    for row_idx, data in enumerate(bug_data, start=1):
        for col_idx, val in enumerate(data):
            t_bugs.rows[row_idx].cells[col_idx].text = val

    doc.add_paragraph("Final Validation Statement:").runs[0].bold = True
    doc.add_paragraph("[X] Yes (The project is 100% complete and ready for final demonstration and evaluation)").runs[0].bold = True

    doc.add_paragraph("b) Output / Result Screenshots:").runs[0].bold = True
    doc.add_paragraph("Screenshot 1: SkinLens AI Main Clinical Diagnostic Dashboard displaying overall scan metrics, benign/malignant counts, prediction probability, decision confidence meter, and recommendations.")
    doc.add_paragraph("Screenshot 2: Explainable AI (Grad-CAM) Visual Attention Overlay Panel displaying original lesion image, gradient activation heatmap, and 0.65 alpha-blended overlay.")
    doc.add_paragraph("Screenshot 3: Automated Hospital-Grade PDF Clinical Screening Report featuring patient demographic table, AI risk card, visual proof pair, clinical disclaimers, and signature block.")

    # 7. Final Results and Outcomes
    h7 = doc.add_heading("7. Final Results and Outcomes", level=2)
    h7.runs[0].font.color.rgb = RGBColor(2, 132, 199)
    doc.add_paragraph("[X] Completed").runs[0].bold = True

    doc.add_paragraph("Completed Components:").runs[0].bold = True
    comp_list = [
        "Complete ISIC 2024 Dataset exploration, cleaning, metadata feature engineering, and patient-wise split pipeline.",
        "Deep Learning classifier based on EfficientNet-B0 trained with PyTorch, Focal Loss, class weighting, AMP, and early stopping.",
        "Threshold optimization via Youden's J statistic for imbalanced medical classification.",
        "Explainable AI (Grad-CAM) module for visual attention heatmap generation.",
        "Multi-heuristic pre-prediction quality gate filter (skin tone ratio + document detector).",
        "Test-Time Augmentation (TTA) multi-view inference engine.",
        "Hospital-grade PDF report generator using ReportLab (pdf_generator.py).",
        "Full-featured Streamlit web application dashboard (app.py) with scan history explorer, checkpoint manager, model metrics inspector, and patient metadata tracking.",
        "Comprehensive system testing, bug resolutions, and complete final project documentation."
    ]
    for item in comp_list:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_paragraph("Pending Components: None (All planned scope 100% completed)").runs[0].bold = True

    # 8. References and Resources Used
    h8 = doc.add_heading("8. References and Resources Used", level=2)
    h8.runs[0].font.color.rgb = RGBColor(2, 132, 199)

    refs = [
        "[1] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of the 36th International Conference on Machine Learning (ICML), 97, 6105–6114.",
        "[2] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 618–626.",
        "[3] Kurtansky, N., Rotemberg, V., Gillis, M., Kose, K., Reade, W., & Chow, A. (2024). ISIC 2024 – Skin Cancer Detection with 3D-TBP. Kaggle. https://www.kaggle.com/competitions/isic-2024-challenge",
        "[4] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems (NeurIPS), 32, 8024–8035.",
        "[5] Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. Information, 11(2), 125.",
        "[6] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press."
    ]
    for ref in refs:
        doc.add_paragraph(ref)

    # Evaluation Table
    doc.add_heading("Evaluation (To be filled by Mentor (Institute))", level=3)
    t_eval = doc.add_table(rows=7, cols=6)
    t_eval.alignment = WD_TABLE_ALIGNMENT.CENTER
    e_headers = ["Criteria", "Excellent", "Good", "Average", "Poor", "Marks"]
    for i, h in enumerate(e_headers):
        t_eval.rows[0].cells[i].text = h
        set_cell_background(t_eval.rows[0].cells[i], "0284C7")
        t_eval.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        t_eval.rows[0].cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    e_data = [
        ("Final Project Completion & Implementation", ">=90% work", "75-89%", "50-74%", "<50%", "/25"),
        ("Functionality, Testing & Validation", "Strong testing", "Good testing", "Limited testing", "No validation", "/25"),
        ("Technical Understanding", "Strong conceptual", "Good understanding", "Basic understanding", "Unable to explain", "/20"),
        ("Project Demonstration & Results", "Complete, clear", "Minor gaps", "Incomplete doc", "Poor doc", "/20"),
        ("References & Citations", "Relevant, proper", "Minor issues", "Limited sources", "Missing/improper", "/10"),
        ("TOTAL", "", "", "", "", "/100")
    ]
    for row_idx, data in enumerate(e_data, start=1):
        for col_idx, val in enumerate(data):
            t_eval.rows[row_idx].cells[col_idx].text = val

    doc.save("Reporting_4_Final_Report.docx")
    print("Reporting_4_Final_Report.docx built successfully!")

if __name__ == "__main__":
    create_docx()
