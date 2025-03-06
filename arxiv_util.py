# arXiv论文爬虫示例
import arxiv
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def search_and_download_arxiv(keyword, dir_path):
    search = arxiv.Search(
    query=keyword,
    max_results=1000,
    sort_by=arxiv.SortCriterion.SubmittedDate
    )
    for result in search.results():
        result.download_pdf(dirpath=dir_path)



def parse_pdf(pdf_path):
    # PDF解析
    elements = partition_pdf("paper.pdf", strategy="auto")
    text = "\n".join([e.text for e in elements])

    # 智能分块（保留图表上下文）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n##", "\n\n", "。", "References"]
    )
    chunks = splitter.split_text(text)
    return chunks

