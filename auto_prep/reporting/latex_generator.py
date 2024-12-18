from typing import List

from pylatex import (
    Center,
    Command,
    Document,
    Figure,
    NoEscape,
    Section,
    Subsection,
    Table,
    Tabular,
)
from pylatex.package import Package
from pylatex.utils import bold

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)
DEFAULT_GEOMETRY: dict = {
    "margin": "0.5in",
    "headheight": "10pt",
    "footskip": "0.2in",
    "tmargin": "0.5in",
    "bmargin": "0.5in",
}


class ReportGenerator:
    """
    Generates LaTeX reports with analysis results and visualizations.
    """

    def __init__(
        self,
        title: str = "Analysis Report",
        geometry_options: dict = DEFAULT_GEOMETRY,
        default_filepath: str = "raport",
    ):
        """
        Args:
            title (str, optional): The title of the report.
                Defaults to "Analysis Report".
            geometry_options (dict, optional): A dictionary of geometry options
                for the LaTeX document. Defaults to DEFAULT_GEOMETRY.
            default_filepath (str): Name of directory to be used for pylatex.
        """
        logger.info("Initializing ReportGenerator with title: %s", title)
        try:
            self.doc = Document(
                default_filepath=default_filepath, geometry_options=geometry_options
            )
            self.title = title

            # Add necessary packages
            self.doc.packages.append(Package("graphicx"))
            self.doc.packages.append(Package("float"))
            self.doc.packages.append(Package("booktabs"))
            self.doc.packages.append(Package("hyperref"))
            self.doc.packages.append(Package("caption"))
            self.doc.packages.append(Package("subcaption"))
            logger.debug("Initialized LaTeX document with packages")
        except Exception as e:
            logger.error("Failed to initialize ReportGenerator: %s", str(e))
            raise

    def add_header(self) -> None:
        """
        Adds the title page and table of contents to the LaTeX document.
        """
        try:
            self.doc.preamble.append(NoEscape(r"\title{" + self.title + "}"))
            self.doc.preamble.append(NoEscape(r"\author{AutoPrep}"))
            self.doc.preamble.append(NoEscape(r"\date{\today}"))
            self.doc.append(NoEscape(r"\maketitle"))
            self.doc.append(
                NoEscape(
                    r"""
                \begin{abstract}
                This raport has been generated with AutoPrep.
                \end{abstract}
            """
                )
            )
            self.doc.append(Command("tableofcontents"))
            self.doc.append(NoEscape(r"\newpage"))
        except Exception as e:
            logger.error("Failed to add header: %s", str(e))
            raise

    def add_section(self, title: str, description: str = "") -> Section:
        """
        Adds a new section to the LaTeX document.

        Args:
            title (str): The title of the section.
            description (str, optional): The description of the section. Defaults to "".

        Returns:
            Section: The newly created section.
        """
        section = Section(title)
        if description:
            section.append(description)
        self.doc.append(section)
        return section

    def add_subsection(self, title: str) -> Subsection:
        """
        Adds a new subsection to the LaTeX document.

        Args:
            title (str): The title of the subsection.

        Returns:
            Subsection: The newly created subsection.
        """
        subsection = Subsection(title)
        self.doc.append(subsection)
        return subsection

    def add_table(
        self,
        data: dict,
        caption: str = None,
        header: List[dict] = ["Category", "Value"],
    ) -> None:
        """Add a table to the document.

        Args:
            data (dict): Dictionary to convert to table.
            caption (str): Table caption. If None, no caption will be set.
            header (List[dict]): Table header. Defaults to ["Category", "Value"].
                If None, no header will be set.
        """
        try:
            with self.doc.create(Table(position="H")) as table:
                with table.create(Center()) as centered:
                    with centered.create(Tabular("l r")) as tabular:
                        if header is not None:
                            tabular.add_hline()
                            tabular.add_row([bold(c) for c in header])
                        tabular.add_hline()
                        for key, value in data.items():
                            if isinstance(value, float):
                                value = f"{value:.4f}"
                            tabular.add_row((str(key), str(value)))
                        tabular.add_hline()
                if caption is not None:
                    table.add_caption(caption)
        except Exception as e:
            logger.error(f"Failed to add table {caption}: {str(e)}")
            raise

    def add_figure(self, path: str, caption: str = None, size: int = 0.9) -> None:
        """Add a figure to the document.

        Args:
            path (str): Path to the figure file.
            caption (str): Figure caption. If None, no caption will be set.
            size (int): % of width image is to take. Defaults to 0.9.
        """
        try:
            with self.doc.create(Figure(position="H")) as fig:
                fig.add_image(path, width=NoEscape(rf"{size}\textwidth"))
                if caption is not None:
                    fig.add_caption(caption)
        except Exception as e:
            logger.error(f"Failed to add figure {path}: {str(e)}")
            raise

    def add_verbatim(self, content: str) -> str:
        """Add verbatim text to the document.

        Args:
            content (str): Text to add in verbatim environment.

        Returns:
            str: Formatted text.
        """
        try:
            self.doc.append(
                NoEscape(
                    r"\begin{verbatim}" + "\n" + content + "\n" + r"\end{verbatim}"
                )
            )
            return content
        except Exception as e:
            logger.error("Failed to add verbatim content: %s", str(e))
            raise

    def generate(self, output_path: str) -> None:
        """Generate the final PDF report.

        Args:
            output_path (str): Path where to save the PDF.
        """
        try:
            logger.debug(f"Generating PDF at {output_path}")
            self.doc.generate_pdf(
                output_path, clean=False, clean_tex=False, compiler="pdflatex"
            )
            self.doc.generate_pdf(output_path, clean_tex=False, compiler="pdflatex")
            logger.info("PDF generation complete")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            raise
