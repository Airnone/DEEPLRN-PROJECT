from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "DEEPLRN_project_revised.pdf"

PAGE_W, PAGE_H = LETTER
MARGIN_X = 0.58 * inch
BOTTOM = 0.55 * inch
TOP = 0.55 * inch
GAP = 0.22 * inch
CONTENT_W = PAGE_W - 2 * MARGIN_X
COL_W = (CONTENT_W - GAP) / 2

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#596273")
BLUE = colors.HexColor("#1F5AA6")
BLUE_LIGHT = colors.HexColor("#EAF2FC")
TEAL = colors.HexColor("#0C7B78")
TEAL_LIGHT = colors.HexColor("#E8F6F4")
ORANGE = colors.HexColor("#B45309")
ORANGE_LIGHT = colors.HexColor("#FFF3E6")
PURPLE = colors.HexColor("#6B3FA0")
PURPLE_LIGHT = colors.HexColor("#F3ECFA")
RULE = colors.HexColor("#CBD2DC")


class AcademicDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=LETTER,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="Document-Level Extraction of Financial Audit Findings from Philippine COA Annual Audit Reports",
            author="Matthew Corral; Eirnan Mykchell O. Dicreto; Miguel Salvador",
            subject="Revised research proposal",
            creator="DEEPLRN Project",
        )

        title_h = 1.62 * inch
        first_body_top = PAGE_H - TOP - title_h - 0.08 * inch
        first_body_h = first_body_top - BOTTOM

        title_frame = Frame(
            MARGIN_X,
            first_body_top + 0.05 * inch,
            CONTENT_W,
            title_h,
            id="title",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        first_left = Frame(
            MARGIN_X,
            BOTTOM,
            COL_W,
            first_body_h,
            id="first-left",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        first_right = Frame(
            MARGIN_X + COL_W + GAP,
            BOTTOM,
            COL_W,
            first_body_h,
            id="first-right",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        later_y = BOTTOM
        later_top = PAGE_H - 0.52 * inch
        later_h = later_top - later_y
        later_left = Frame(
            MARGIN_X,
            later_y,
            COL_W,
            later_h,
            id="later-left",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        later_right = Frame(
            MARGIN_X + COL_W + GAP,
            later_y,
            COL_W,
            later_h,
            id="later-right",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )

        self.addPageTemplates(
            [
                PageTemplate(
                    id="First",
                    frames=[title_frame, first_left, first_right],
                    onPage=self._draw_first_page,
                ),
                PageTemplate(
                    id="Later",
                    frames=[later_left, later_right],
                    onPage=self._draw_later_page,
                ),
            ]
        )

    @staticmethod
    def _draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        canvas.line(MARGIN_X, 0.39 * inch, PAGE_W - MARGIN_X, 0.39 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(PAGE_W / 2, 0.22 * inch, str(doc.page))
        canvas.restoreState()

    def _draw_first_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BLUE)
        canvas.rect(0, PAGE_H - 0.12 * inch, PAGE_W, 0.12 * inch, stroke=0, fill=1)
        canvas.restoreState()
        self._draw_footer(canvas, doc)

    def _draw_later_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(MUTED)
        left = "DEEPLRN RESEARCH PROPOSAL"
        right = "CORRAL, DICRETO, AND SALVADOR"
        canvas.drawString(MARGIN_X, PAGE_H - 0.31 * inch, left)
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 0.31 * inch, right)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.45)
        canvas.line(MARGIN_X, PAGE_H - 0.38 * inch, PAGE_W - MARGIN_X, PAGE_H - 0.38 * inch)
        canvas.restoreState()
        self._draw_footer(canvas, doc)


class ArchitectureDiagram(Flowable):
    def __init__(self):
        super().__init__()
        self.width = COL_W
        self.height = 224

    def wrap(self, avail_width, avail_height):
        self.width = min(avail_width, COL_W)
        return self.width, self.height

    def _box(self, canvas, x, y, w, h, fill, stroke, title, subtitle="", title_size=7.0):
        canvas.setFillColor(fill)
        canvas.setStrokeColor(stroke)
        canvas.setLineWidth(0.8)
        canvas.roundRect(x, y, w, h, 4, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", title_size)
        canvas.drawCentredString(x + w / 2, y + h - 12, title)
        if subtitle:
            canvas.setFont("Helvetica", 5.9)
            canvas.setFillColor(MUTED)
            max_width = w - 8
            words = subtitle.split()
            lines = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if stringWidth(candidate, "Helvetica", 5.9) <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            base = y + h - 22
            for line in lines[:2]:
                canvas.drawCentredString(x + w / 2, base, line)
                base -= 7

    @staticmethod
    def _arrow(canvas, x, y1, y2):
        canvas.setStrokeColor(MUTED)
        canvas.setFillColor(MUTED)
        canvas.setLineWidth(0.8)
        canvas.line(x, y1, x, y2)
        canvas.line(x, y2, x - 2.4, y2 + 4)
        canvas.line(x, y2, x + 2.4, y2 + 4)

    def draw(self):
        c = self.canv
        w = self.width
        center = w / 2
        main_w = w * 0.68
        x = center - main_w / 2

        self._box(c, x, 184, main_w, 35, BLUE_LIGHT, BLUE, "COA REPORT INGESTION", "PDF text, OCR fallback, page evidence")
        self._arrow(c, center, 184, 177)
        self._box(c, x, 142, main_w, 35, TEAL_LIGHT, TEAL, "PREPROCESSING", "384-token chunks, 64-token overlap")
        self._arrow(c, center, 142, 135)
        self._box(c, x, 100, main_w, 35, PURPLE_LIGHT, PURPLE, "SHARED REPRESENTATION", "RoBERTa plus 2-layer document Transformer")

        c.setStrokeColor(TEAL)
        c.setDash(2, 2)
        c.line(x + main_w, 159, w - 4, 159)
        c.setDash()
        c.setFillColor(TEAL)
        c.setFont("Helvetica", 5.3)
        c.drawRightString(w - 4, 169, "PLANNED ABLATION")
        c.drawRightString(w - 4, 162, "page + section + bbox")

        self._arrow(c, center, 100, 92)
        gap = 4
        head_w = (w - 2 * gap) / 3
        titles = [("NER", "8 entity types"), ("FINDING", "5 categories"), ("RELATIONS", "evidence-linked")]
        for idx, (title, subtitle) in enumerate(titles):
            hx = idx * (head_w + gap)
            self._box(c, hx, 53, head_w, 39, ORANGE_LIGHT, ORANGE, title, subtitle, title_size=6.5)

        for hx in (head_w / 2, head_w + gap + head_w / 2, 2 * (head_w + gap) + head_w / 2):
            c.setStrokeColor(MUTED)
            c.line(center, 100, hx, 92)

        self._arrow(c, center, 53, 45)
        self._box(c, x, 8, main_w, 37, BLUE_LIGHT, BLUE, "REVIEW OUTPUT", "JSON tuples, confidence, source-page evidence")


def make_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "PaperTitle",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=18.5,
        leading=20.5,
        textColor=INK,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    styles["authors"] = ParagraphStyle(
        "Authors",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=12,
        textColor=INK,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    styles["affiliation"] = ParagraphStyle(
        "Affiliation",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=9.3,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
    styles["body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Times-Roman",
        fontSize=8.35,
        leading=10.25,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=4.2,
        allowWidows=0,
        allowOrphans=0,
        splitLongWords=True,
    )
    styles["abstract"] = ParagraphStyle(
        "AbstractBody",
        parent=styles["body"],
        fontSize=8.1,
        leading=9.9,
        leftIndent=5,
        rightIndent=5,
        borderColor=BLUE,
        borderWidth=0.8,
        borderPadding=(5, 7, 5, 7),
        backColor=colors.HexColor("#F7FAFE"),
        spaceAfter=5,
    )
    styles["section"] = ParagraphStyle(
        "Section",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11.0,
        leading=12.5,
        textColor=BLUE,
        spaceBefore=7.5,
        spaceAfter=3.5,
        keepWithNext=True,
    )
    styles["subsection"] = ParagraphStyle(
        "Subsection",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=10.2,
        textColor=INK,
        spaceBefore=5.5,
        spaceAfter=2,
        keepWithNext=True,
    )
    styles["keywords"] = ParagraphStyle(
        "Keywords",
        parent=styles["body"],
        fontSize=7.6,
        leading=9.1,
        textColor=MUTED,
        spaceAfter=5,
    )
    styles["caption"] = ParagraphStyle(
        "Caption",
        parent=styles["body"],
        fontName="Times-Italic",
        fontSize=7.1,
        leading=8.5,
        alignment=TA_LEFT,
        spaceBefore=2,
        spaceAfter=5,
    )
    styles["reference"] = ParagraphStyle(
        "Reference",
        parent=styles["body"],
        fontSize=6.65,
        leading=7.8,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2.2,
        alignment=TA_LEFT,
        splitLongWords=True,
    )
    styles["note"] = ParagraphStyle(
        "Note",
        parent=styles["body"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.6,
        textColor=MUTED,
        borderColor=RULE,
        borderWidth=0.5,
        borderPadding=5,
        backColor=colors.HexColor("#F8F9FB"),
        spaceBefore=3,
        spaceAfter=5,
    )
    return styles


def p(text, styles, style="body"):
    return Paragraph(text, styles[style])


def bullets(items, styles):
    return ListFlowable(
        [ListItem(p(item, styles), leftIndent=8) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
        bulletFontName="Helvetica",
        bulletFontSize=5.5,
        bulletOffsetY=1.5,
        spaceAfter=4,
    )


def build_story(styles):
    story = [
        Spacer(1, 5),
        p("Document-Level Extraction of Financial Audit Findings from Philippine COA Annual Audit Reports", styles, "title"),
        p("Matthew Corral &nbsp;&nbsp;|&nbsp;&nbsp; Eirnan Mykchell O. Dicreto &nbsp;&nbsp;|&nbsp;&nbsp; Miguel Salvador", styles, "authors"),
        p("De La Salle University, Manila, Philippines", styles, "affiliation"),
        p("matthew_corrall@dlsu.edu.ph &nbsp;&nbsp; eirnan_dicreto@dlsu.edu.ph &nbsp;&nbsp; migs_salvador@dlsu.edu.ph", styles, "affiliation"),
        p("RESEARCH PROPOSAL - AUGUST 2026", styles, "affiliation"),
        NextPageTemplate("Later"),
        FrameBreak(),
        p("ABSTRACT", styles, "section"),
        p(
            "The Philippine Commission on Audit (COA) publishes annual audit reports containing financial statements, audit observations, and recommendations. Locating related findings, monetary amounts, organizations, projects, and source evidence across long and inconsistently formatted reports is labor intensive. This proposal evaluates whether a document-level, multi-task Transformer can extract evidence-linked audit-finding tuples more accurately than keyword, sentence-level, long-context, and layout-aware baselines. The system combines PDF extraction and OCR fallback, overlapping RoBERTa chunks, cross-chunk attention, named-entity recognition, finding classification, and relation extraction. Evaluation will use leakage-resistant LGU splits, a documented annotation protocol, an entirely human-labeled validation and test set, cluster-bootstrap confidence intervals, and paired significance tests. Outputs will be presented as review aids with confidence and page-level evidence; they will not determine misconduct, legal liability, or intent.",
            styles,
            "abstract",
        ),
        p("<b>Keywords:</b> document understanding; named-entity recognition; relation extraction; financial auditing; Philippine government; human-in-the-loop review", styles, "keywords"),

        p("1. INTRODUCTION", styles, "section"),
        p(
            "COA annual audit reports are final products of yearly audits and include audit observations and recommendations [1]. A report may span hundreds of pages and distribute a single finding across narrative text, tables, appendices, and repeated references. Human reviewers must often connect an observation in one section with a peso amount, an office or contractor, and supporting evidence elsewhere in the document. This structure makes cross-report comparison slow and limits the practical use of a large public corpus.",
            styles,
        ),
        p(
            "This project asks: <i>Can a document-level Transformer extract evidence-linked audit findings, amounts, and associated entities from COA annual audit reports more accurately than keyword, sentence-level, long-context, and layout-aware baselines?</i> The target output is a reviewable tuple, not a judgment of misconduct. Terms such as <i>finding</i> and <i>associated entity</i> refer only to what is stated in the report and do not establish intent, guilt, or legal responsibility.",
            styles,
        ),
        p("The proposed contributions are:", styles),
        bullets(
            [
                "a documented and quality-controlled corpus of COA annual audit reports;",
                "a multi-task model for entity, finding-category, and evidence-linked relation extraction;",
                "leakage-resistant evaluation across held-out LGUs and a secondary temporal stress test; and",
                "evidence-linked JSON output that preserves confidence and source-page provenance for downstream review.",
            ],
            styles,
        ),

        p("2. RELATED WORK", styles, "section"),
        p(
            "BERT established bidirectional Transformer pre-training for token and sentence tasks [2], while RoBERTa improved its training procedure [3]. Longformer extends attention to long sequences [4], and hierarchical Transformers aggregate representations across document segments [17]. These approaches motivate a model that encodes local text while also reasoning across chunks.",
            styles,
        ),
        p(
            "Document layout is also relevant. LayoutLM combines text with two-dimensional position [5], LayoutLMv2 adds spatial and visual interactions [6], and LayoutLMv3 unifies text and image masking [7]. DocRED demonstrates that relation extraction frequently requires cross-sentence evidence [8]. Joint entity and relation models further show that shared learning can reduce pipeline error propagation [9, 10]. Financial language and OCR introduce additional domain shift [11, 13], while neural NER provides a foundation for extracting names and monetary expressions [12]. No cited work directly evaluates evidence-linked finding tuples in long Philippine COA annual audit reports.",
            styles,
        ),

        p("3. PROPOSED METHODOLOGY", styles, "section"),
        p("3.1 Corpus and preprocessing", styles, "subsection"),
        p(
            "We will sample approximately 1,000 searchable COA annual audit report PDFs from 2018-2023, covering at least 100 LGUs through the official COA repository [1]. The sampling log will record URL, LGU, year, report type, download date, file hash, page count, and extraction status. PDF text, word bounding boxes, page identifiers, headings, and tables will be retained. Pages with fewer than 30 extractable characters will trigger 300-dpi Tesseract OCR. Cleaning will normalize spacing and currency symbols while preserving offsets to the source page.",
            styles,
        ),
        p(
            "Reports will be segmented into sentence-aware 384-token RoBERTa windows with a 64-token overlap. Exact hashes and MinHash-style near-duplicate detection will cluster repeated paragraphs before partitioning. Duplicate clusters will never cross train, validation, or test boundaries.",
            styles,
        ),
        KeepTogether(
            [
                ArchitectureDiagram(),
                p("<b>Figure 1.</b> Revised architecture. The implemented text prototype supplies the core shared representation; page, section, and bounding-box fusion is a planned, separately ablated extension. Every predicted tuple retains source-page evidence.", styles, "caption"),
            ]
        ),
        p("3.2 Model architecture", styles, "subsection"),
        p(
            "RoBERTa-base will encode each chunk. A two-layer, eight-head document Transformer with learned chunk-position embeddings will contextualize chunk representations. The enriched chunk state will be fused with token embeddings for BIO sequence labeling. Three supervised heads will be optimized jointly: (1) NER for PERSON, ORGANIZATION, LGU, CONTRACTOR, AMOUNT, DATE, PROJECT, and FINDING; (2) classification into unauthorized expenditure, unliquidated cash advance, procurement irregularity, unsupported disbursement, and contractor-related concern; and (3) bilinear relation scoring for INVOLVES, AMOUNT_OF, and ASSOCIATED_WITH, plus NO_RELATION.",
            styles,
        ),
        p(
            "The current repository implements RoBERTa chunk encoding, cross-chunk attention, BIO tagging, document classification, and bilinear relation scoring. A planned layout-aware extension will concatenate learned page, section-type, and normalized bounding-box embeddings with token states and project them back to the encoder dimension. Because this fusion is not yet implemented, it will be reported as an extension and evaluated through a dedicated ablation rather than described as part of the existing prototype.",
            styles,
        ),
        p(
            "Candidate relations will be limited to entities no more than 12 sentences apart during the initial experiment. This threshold will be tuned only on the validation set and compared with 6-, 18-, and unrestricted-sentence alternatives. The decoder will return JSON containing document metadata, entities, finding categories, relations, calibrated confidence, and the exact page and sentence evidence used for each prediction.",
            styles,
        ),

        p("3.3 Annotation and label quality", styles, "subsection"),
        p(
            "At least 250 reports will receive human annotation. A pilot of 20 reports will refine the guidelines before full labeling. The manual will define entity boundaries, finding categories, cross-page relations, table handling, negated or resolved observations, uncertainty language, and the distinction between textual association and legal responsibility. Two annotators will independently label every validation and test document and at least 20% of the training documents. Disagreements will be adjudicated by consensus or a third reviewer.",
            styles,
        ),
        p(
            "We will report exact-span agreement for NER, Cohen's kappa for finding categories, and pairwise relation F1. Rule-based date and amount detection and weak supervision [15] may create additional training examples, but no weak label will enter validation or testing. The final validation and test partitions will be entirely human labeled.",
            styles,
        ),

        p("3.4 Leakage-resistant partitioning", styles, "subsection"),
        p(
            "The primary split will assign whole LGUs, not LGU-year pairs, to partitions: approximately 70% of LGUs for training, 15% for validation, and 15% for testing. All years belonging to one LGU will remain in the same partition. This prevents recurring template language from the same institution from appearing in both training and testing. A secondary temporal stress test will train on 2018-2021 reports and evaluate on 2022-2023 reports, using previously unseen LGUs when the sample size permits.",
            styles,
        ),

        p("4. EXPERIMENTAL PLAN", styles, "section"),
        p("4.1 Baselines and ablations", styles, "subsection"),
        p(
            "The system will be compared with: (B1) TF-IDF plus a linear SVM [14]; (B2) sentence-level RoBERTa; (B3) Longformer without page-layout features; and (B4) LayoutLMv3 [7]. Hyperparameters will be selected using validation tuple F1 only. Each neural model will be trained with five fixed random seeds. Ablations will remove cross-chunk attention, layout fusion, OCR fallback, weakly supervised pre-training, and individual multi-task heads.",
            styles,
        ),
        p("4.2 Metrics and statistical analysis", styles, "subsection"),
        p(
            "The preregistered primary outcome will be exact-match micro-F1 for complete FINDING-AMOUNT-ENTITY tuples. Secondary outcomes will include NER precision, recall, and exact-span F1 [16]; relation F1; macro-F1 across LGUs; per-category scores; runtime; and evidence-page accuracy. Calibration will be measured with the Brier score and expected calibration error.",
            styles,
        ),
        p(
            "We will report the mean and standard deviation across five seeds. Ninety-five percent confidence intervals will use cluster bootstrap resampling at the LGU level so that findings from the same institution are not treated as independent. Each baseline comparison will use a paired cluster-bootstrap or permutation test with Holm correction for multiple comparisons. The project will claim an improvement only if the strongest-baseline gain is statistically significant at alpha = 0.05 and at least 3 absolute tuple-F1 points.",
            styles,
        ),
        p("4.3 Error analysis", styles, "subsection"),
        p(
            "Errors will be categorized by OCR failure, table structure, entity boundary, finding-category ambiguity, cross-page distance, near-duplicate language, and relation direction. Results will be stratified by LGU, year, report length, and OCR status. This analysis will distinguish model limitations from extraction and annotation failures.",
            styles,
        ),

        p("5. RISKS, ETHICS, AND REPRODUCIBILITY", styles, "section"),
        p(
            "Although COA reports are public, extracted names may appear near adverse financial observations. The output schema will therefore use neutral labels, preserve uncertainty and negation, and attach source evidence to every exported tuple. The system will not rank people by suspected misconduct, infer intent, or make legal conclusions. False-positive and subgroup error rates will be documented, and each exported prediction will retain its model version, confidence, and evidence provenance.",
            styles,
        ),
        p(
            "Subject to COA redistribution rules, we will release code, annotation guidelines, file hashes, split manifests, random seeds, model configurations, and evaluation scripts. If report redistribution is not permitted, the release will provide URLs, hashes, and reproducible download instructions. The original PDFs will remain the authoritative evidence source.",
            styles,
        ),

        p("6. EXPECTED OUTCOME", styles, "section"),
        p(
            "The project will determine whether document-level, multi-task modeling improves extraction of reviewable financial audit findings from long COA reports. Success requires a reproducible gain over the strongest baseline under held-out-LGU evaluation, reliable evidence links, and useful uncertainty estimates. Regardless of the result, the annotation study and error analysis will document the linguistic, layout, and OCR challenges that must be solved before broader deployment.",
            styles,
        ),
        p(
            "The proposal deliberately does not claim to detect systemic misappropriation. A later study could aggregate validated findings across LGUs and years to analyze recurrence, contractor networks, or temporal trends, but those analyses require separate definitions, denominators, and human validation.",
            styles,
            "note",
        ),

        p("REFERENCES", styles, "section"),
    ]

    references = [
        "[1] Commission on Audit, Republic of the Philippines. <link href='https://www.coa.gov.ph/reports/annual-audit-reports/aar-local-government-units/' color='#1F5AA6'>Annual Audit Reports: Local Government Units</link>. Accessed 7 August 2026.",
        "[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. <i>Proceedings of NAACL-HLT</i>, 4171-4186. doi:10.18653/v1/N19-1423.",
        "[3] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.",
        "[4] Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The Long-Document Transformer. arXiv:2004.05150.",
        "[5] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. <i>Proceedings of KDD</i>, 1192-1200. doi:10.1145/3394486.3403172.",
        "[6] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. 2021. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. <i>Proceedings of ACL-IJCNLP</i>, 2579-2591. doi:10.18653/v1/2021.acl-long.201.",
        "[7] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. arXiv:2204.08387.",
        "[8] Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, and Maosong Sun. 2019. DocRED: A Large-Scale Document-Level Relation Extraction Dataset. <i>Proceedings of ACL</i>, 764-777. doi:10.18653/v1/P19-1074.",
        "[9] Giannis Bekoulis, Johannes Deleu, Thomas Demeester, and Chris Develder. 2018. Joint Entity Recognition and Relation Extraction as a Multi-head Selection Problem. <i>Expert Systems with Applications</i> 114, 34-45. doi:10.1016/j.eswa.2018.07.032.",
        "[10] Markus Eberts and Adrian Ulges. 2020. Span-based Joint Entity and Relation Extraction with Transformer Pre-training. <i>Proceedings of ECAI</i>, 2006-2013.",
        "[11] Zhuang Liu, Duyu Huang, Kai Huang, Zhuang Li, and Jun Zhao. 2020. FinBERT: A Pre-trained Financial Language Representation Model for Financial Text Mining. <i>Proceedings of IJCAI</i>, 4513-4519. doi:10.24963/ijcai.2020/622.",
        "[12] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural Architectures for Named Entity Recognition. <i>Proceedings of NAACL-HLT</i>, 260-270. doi:10.18653/v1/N16-1030.",
        "[13] Ahmed Hamdi, Antoine Jean-Caurant, Nicolas Sidere, Mickael Coustaty, and Antoine Doucet. 2021. The Impact of OCR Quality on Named Entity Recognition. <i>Proceedings of TPDL</i>, 22-35.",
        "[14] Thorsten Joachims. 1998. Text Categorization with Support Vector Machines: Learning with Many Relevant Features. <i>Proceedings of ECML</i>, 137-142.",
        "[15] Alexander Ratner, Stephen H. Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Re. 2017. Snorkel: Rapid Training Data Creation with Weak Supervision. <i>Proceedings of the VLDB Endowment</i> 11(3), 269-282.",
        "[16] Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 Shared Task: Language-independent Named Entity Recognition. <i>Proceedings of CoNLL</i>, 142-147.",
        "[17] Raghavendra Pappagari, Piotr Zelasko, Jesus Villalba, Yishay Carmiel, and Najim Dehak. 2019. Hierarchical Transformers for Long Document Classification. <i>Proceedings of ASRU</i>, 838-844.",
    ]
    for index, ref in enumerate(references, 1):
        story.append(p(ref, styles, "reference"))
        if index == 10:
            story.append(FrameBreak())
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = AcademicDocTemplate(str(OUTPUT))
    doc.build(build_story(styles))
    print(OUTPUT)


if __name__ == "__main__":
    main()
