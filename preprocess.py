import os
import glob
import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path
import pdfplumber

# 工业级PDF批量处理器，生产一线级别的代码 [cite: 5173, 5174]
class PDFBatchProcessor:
    def __init__(self, output_dir: str = "./pdf_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # 配置日志系统 [cite: 5178, 5179]
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "pdf_processing.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    # 查找指定路径下的所有PDF文件 [cite: 5189]
    def find_pdf_files(self, input_path: str) -> List[Path]:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == '.pdf':
            return [path]
        elif path.is_dir():
            # 递归查找所有PDF文件 [cite: 5195]
            pdf_files = list(path.glob("**/*.pdf"))
            self.logger.info(f"在 {input_path} 中找到 {len(pdf_files)} 个PDF文件")
            return pdf_files
        else:
            raise ValueError(f"路径不存在，或不是PDF文件: {input_path}")

    # 提取单个PDF文件的内容 [cite: 5201]
    def extract_pdf_content(self, 
                           pdf_path: Path, 
                           extract_text: bool = True, 
                           extract_tables: bool = True, 
                           table_settings: Optional[dict] = None) -> Dict:
        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "metadata": {},
            "pages": [],
            "error": None
        }
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 提取元数据 [cite: 5225]
                result["metadata"] = pdf.metadata
                for page_num, page in enumerate(pdf.pages, 1):
                    page_result = {"page_number": page_num, "text": "", "tables": []}
                    # 提取文本 [cite: 5229]
                    if extract_text:
                        try:
                            # 布局模式根据需求调整 [cite: 5231, 5232]
                            text = page.extract_text(layout=False)
                            page_result["text"] = text if text else ""
                        except Exception as e:
                            self.logger.warning(f"页面 {page_num} 文本提取失败: {str(e)}")
                    # 提取表格 [cite: 5238, 5240]
                    if extract_tables:
                        try:
                            tables = page.extract_tables(table_settings or {})
                            if tables:
                                page_result["tables"] = tables
                        except Exception as e:
                            self.logger.warning(f"页面 {page_num} 表格提取失败: {str(e)}")
                    
                    result["pages"].append(page_result)
                
                self.logger.info(f"成功处理: {pdf_path.name} ({len(pdf.pages)}页)")
        except Exception as e:
            error_msg = f"处理文件失败 {pdf_path}: {str(e)}"
            result["error"] = error_msg
            self.logger.error(error_msg)
        return result

    # 批量处理PDF文件 [cite: 5259]
    def process_batch(self, pdf_files: List[Path], save_format: str = "excel", **extract_kwargs) -> pd.DataFrame:
        all_results = []
        for i, pdf_file in tqdm(enumerate(pdf_files, 1), total=len(pdf_files)):
            self.logger.info(f"处理进度: {i}/{len(pdf_files)} {pdf_file.name}")
            result = self.extract_pdf_content(pdf_file, **extract_kwargs)
            all_results.append(result)
            
            # 实时保存进度(针对大批量处理) [cite: 5272, 5273]
            if i % 10 == 0:
                self._save_intermediate_results(all_results, f"batch_{i}")
        
        return self._save_results(all_results, save_format)

    # 保存最终处理结果 [cite: 5278]
    def _save_results(self, results: List[Dict], format: str) -> pd.DataFrame:
        flat_data = []
        detailed_results = []
        
        for result in results:
            if result["error"]:
                flat_data.append({
                    "file_name": result["file_name"],
                    "status": "Error",
                    "error_message": result["error"],
                    "page_count": 0,
                    "text_length": 0,
                    "table_count": 0
                })
                continue

            total_text = ""
            total_tables = 0
            for page in result["pages"]:
                total_text += page["text"]
                total_tables += len(page["tables"])
                # 同时准备详细文本内容用于 Excel [cite: 5313, 5319]
                if page["text"]:
                    detailed_results.append({
                        "file_name": result["file_name"],
                        "page_number": page["page_number"],
                        "text_content": page["text"]
                    })

            flat_data.append({
                "file_name": result["file_name"],
                "status": "Success",
                "error_message": "",
                "page_count": len(result["pages"]),
                "text_length": len(total_text),
                "table_count": total_tables,
                "author": result["metadata"].get("Author", ""),
                "creation_date": result["metadata"].get("CreationDate", "")
            })

        df = pd.DataFrame(flat_data)
        
        # 根据格式保存 [cite: 5311, 5325, 5329]
        if format.lower() == "excel":
            df.to_excel(self.output_dir / "pdf_extraction_summary.xlsx", index=False)
            if detailed_results:
                pd.DataFrame(detailed_results).to_excel(
                    self.output_dir / "pdf_detailed_text.xlsx", index=False
                )
        elif format.lower() == "csv":
            df.to_csv(self.output_dir / "pdf_extraction_summary.csv", index=False)
            
        self.logger.info(f"结果已保存到 {self.output_dir}")
        return df

    # 保存中间进度 [cite: 5333, 5335]
    def _save_intermediate_results(self, results: List[Dict], batch_name: str):
        try:
            temp_df = pd.DataFrame([{
                "file_name": r["file_name"],
                "status": "Error" if r["error"] else "Success",
                "pages_processed": len(r["pages"])
            } for r in results])
            temp_df.to_csv(self.output_dir / f"progress_{batch_name}.csv", index=False)
        except Exception as e:
            self.logger.warning(f"保存中间结果失败: {str(e)}")

# 高级表格提取配置 [cite: 5345]
ADVANCED_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 10,
    "edge_min_length": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1
}

def main():
    # 实例化PDF处理器对象 [cite: 5356]
    # 注意：确保您的 PDF 文件存放在项目根目录下的 pdf_documents 文件夹中 [cite: 5360]
    processor = PDFBatchProcessor(output_dir="./pdf_output")
    try:
        # 查找PDF文件 [cite: 5359, 5360]
        pdf_files = processor.find_pdf_files("./pdf_documents")
        if not pdf_files:
            processor.logger.warning("未找到PDF文件")
            return
        
        # 批量处理 [cite: 5365]
        results_df = processor.process_batch(
            pdf_files,
            save_format="excel",
            extract_text=True,
            extract_tables=True,
            table_settings=ADVANCED_TABLE_SETTINGS
        )
        
        # 打印摘要统计 [cite: 5373]
        success_count = len(results_df[results_df["status"] == "Success"])
        processor.logger.info(f"处理完成: {success_count}/{len(pdf_files)} 个文件成功")
        
    except Exception as e:
        # 处理过程中发生错误，记录日志 [cite: 5382]
        if 'processor' in locals():
            processor.logger.error(f"处理过程发生错误: {str(e)}")
        else:
            print(f"初始化失败: {str(e)}")

if __name__ == "__main__":
    main()