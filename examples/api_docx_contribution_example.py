"""
Example script demonstrating how to contribute Word documents via the API.

This script shows how to use the knowledge contribution API endpoint
to upload and process DOCX files.
"""

import requests
import json
from pathlib import Path


# API Configuration
API_BASE_URL = "http://localhost:8000/api"
KNOWLEDGE_ENDPOINT = f"{API_BASE_URL}/knowledge/contribute"


def contribute_docx_file(
    docx_file_path: str,
    title: str,
    content: str,
    meta_tags: str,
    source: str = None,
    author_name: str = None,
    additional_references: str = None,
    is_public: bool = False
):
    """
    Contribute a DOCX file to the knowledge base via the API.
    
    Args:
        docx_file_path: Path to the DOCX file
        title: Title of the knowledge entry
        content: Description or summary of the content
        meta_tags: Comma-separated tags (e.g., "agriculture,wheat,fertilizer")
        source: Source/origin of the knowledge (optional)
        author_name: Name of the contributor (optional)
        additional_references: Additional references or URLs (optional)
        is_public: Whether this is public information
    
    Returns:
        Response from the API
    """
    
    # Prepare form data
    data = {
        'title': title,
        'content': content,
        'meta_tags': meta_tags,
        'is_public': is_public
    }
    
    # Add optional fields if provided
    if source:
        data['source'] = source
    if author_name:
        data['author_name'] = author_name
    if additional_references:
        data['additional_references'] = additional_references
    
    # Prepare file for upload
    files = None
    if docx_file_path and Path(docx_file_path).exists():
        files = {
            'file': (
                Path(docx_file_path).name,
                open(docx_file_path, 'rb'),
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        }
    
    try:
        # Send POST request
        response = requests.post(
            KNOWLEDGE_ENDPOINT,
            data=data,
            files=files
        )
        
        # Close file if it was opened
        if files:
            files['file'][1].close()
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'data': result,
                'message': 'DOCX file contributed successfully!'
            }
        else:
            return {
                'success': False,
                'error': response.text,
                'status_code': response.status_code
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def example_1_basic_docx_contribution():
    """Example 1: Basic DOCX file contribution."""
    print("\n" + "="*70)
    print("Example 1: Basic DOCX File Contribution")
    print("="*70)
    
    result = contribute_docx_file(
        docx_file_path="docs/agricultural_guide.docx",
        title="کشاورزی پایدار",  # Sustainable Agriculture
        content="راهنمای جامع کشاورزی پایدار شامل تکنیک‌های مدرن و سنتی",
        meta_tags="agriculture,sustainable,guide,persian",
        source="Agricultural Research Institute",
        author_name="John Doe",
        is_public=True
    )
    
    if result['success']:
        print("✓ Success!")
        print(f"  Contribution ID: {result['data'].get('contribution', {}).get('id')}")
        print(f"  Title: {result['data'].get('contribution', {}).get('title')}")
        print(f"  File Processed: {result['data'].get('contribution', {}).get('file_processed')}")
        print(f"  File Type: {result['data'].get('contribution', {}).get('file_type')}")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_2_text_only_contribution():
    """Example 2: Text-only contribution (no file)."""
    print("\n" + "="*70)
    print("Example 2: Text-Only Contribution (No File)")
    print("="*70)
    
    result = contribute_docx_file(
        docx_file_path=None,  # No file
        title="نکات کلیدی آبیاری",  # Key Irrigation Tips
        content="آبیاری قطره‌ای یکی از روش‌های مؤثر برای صرفه‌جویی در مصرف آب است",
        meta_tags="irrigation,water,efficiency",
        author_name="Agricultural Expert",
        is_public=True
    )
    
    if result['success']:
        print("✓ Success!")
        print(f"  Contribution ID: {result['data'].get('contribution', {}).get('id')}")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_3_multiple_file_contributions():
    """Example 3: Contribute multiple DOCX files."""
    print("\n" + "="*70)
    print("Example 3: Multiple DOCX File Contributions")
    print("="*70)
    
    files_to_contribute = [
        {
            'path': 'docs/wheat_cultivation.docx',
            'title': 'کشت گندم',
            'content': 'راهنمای کامل کشت و برداشت گندم',
            'meta_tags': 'wheat,cultivation,grain',
        },
        {
            'path': 'docs/soil_management.docx',
            'title': 'مدیریت خاک',
            'content': 'تکنیک‌های بهبود کیفیت خاک',
            'meta_tags': 'soil,management,quality',
        },
        {
            'path': 'docs/pest_control.docx',
            'title': 'کنترل آفات',
            'content': 'روش‌های ارگانیک کنترل آفات',
            'meta_tags': 'pest,control,organic',
        }
    ]
    
    results = []
    for file_info in files_to_contribute:
        if Path(file_info['path']).exists():
            result = contribute_docx_file(
                docx_file_path=file_info['path'],
                title=file_info['title'],
                content=file_info['content'],
                meta_tags=file_info['meta_tags'],
                is_public=True
            )
            results.append({
                'file': file_info['path'],
                'success': result['success']
            })
            print(f"  {'✓' if result['success'] else '✗'} {Path(file_info['path']).name}")
        else:
            print(f"  ⚠ File not found: {file_info['path']}")
    
    print(f"\n  Total: {len(results)} files processed")
    print(f"  Success: {sum(1 for r in results if r['success'])}")
    print(f"  Failed: {sum(1 for r in results if not r['success'])}")


def example_4_mixed_file_types():
    """Example 4: Contribute different file types."""
    print("\n" + "="*70)
    print("Example 4: Mixed File Type Contributions")
    print("="*70)
    
    contributions = [
        {
            'type': 'DOCX',
            'path': 'docs/report.docx',
            'title': 'گزارش سالانه کشاورزی',
            'meta_tags': 'report,annual,agriculture'
        },
        {
            'type': 'PDF',
            'path': 'docs/research.pdf',
            'title': 'تحقیقات علمی',
            'meta_tags': 'research,scientific'
        },
        {
            'type': 'Excel',
            'path': 'docs/qa_data.xlsx',
            'title': 'سوالات متداول',
            'meta_tags': 'faq,questions'
        }
    ]
    
    for contrib in contributions:
        if Path(contrib['path']).exists():
            result = contribute_docx_file(
                docx_file_path=contrib['path'],
                title=contrib['title'],
                content=f"File type: {contrib['type']}",
                meta_tags=contrib['meta_tags'],
                is_public=True
            )
            status = '✓' if result['success'] else '✗'
            print(f"  {status} {contrib['type']}: {Path(contrib['path']).name}")
        else:
            print(f"  ⚠ File not found: {contrib['path']}")


def example_5_curl_examples():
    """Example 5: Print curl command examples."""
    print("\n" + "="*70)
    print("Example 5: CURL Command Examples")
    print("="*70)
    
    print("\nContribute DOCX file:")
    print("""
curl -X POST "http://localhost:8000/api/knowledge/contribute" \\
  -F "title=کشاورزی پایدار" \\
  -F "content=راهنمای جامع کشاورزی" \\
  -F "meta_tags=agriculture,guide,persian" \\
  -F "author_name=John Doe" \\
  -F "is_public=true" \\
  -F "file=@docs/agricultural_guide.docx"
    """)
    
    print("\nContribute without file:")
    print("""
curl -X POST "http://localhost:8000/api/knowledge/contribute" \\
  -F "title=نکات کشاورزی" \\
  -F "content=نکات مفید برای کشاورزان" \\
  -F "meta_tags=tips,farming" \\
  -F "is_public=true"
    """)
    
    print("\nContribute with all file types:")
    print("""
# DOCX
curl -X POST "http://localhost:8000/api/knowledge/contribute" \\
  -F "title=Word Document" \\
  -F "content=Description" \\
  -F "meta_tags=docx,word" \\
  -F "file=@document.docx"

# PDF
curl -X POST "http://localhost:8000/api/knowledge/contribute" \\
  -F "title=PDF Document" \\
  -F "content=Description" \\
  -F "meta_tags=pdf" \\
  -F "file=@document.pdf"

# Excel
curl -X POST "http://localhost:8000/api/knowledge/contribute" \\
  -F "title=Excel QA" \\
  -F "content=Description" \\
  -F "meta_tags=excel,qa" \\
  -F "file=@questions.xlsx"
    """)


def main():
    """Run all examples."""
    print("="*70)
    print("DOCX Knowledge Contribution API Examples")
    print("="*70)
    print("\nNote: Make sure the API server is running at", API_BASE_URL)
    print("Start server with: uvicorn main:app --reload")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api', '')}/health")
        if response.status_code == 200:
            print("✓ API server is running\n")
        else:
            print("⚠ API server may not be running properly\n")
    except:
        print("✗ API server is not running. Please start it first.\n")
        return
    
    # Run examples
    examples = [
        ("Basic DOCX Contribution", example_1_basic_docx_contribution),
        ("Text-Only Contribution", example_2_text_only_contribution),
        ("Multiple Files", example_3_multiple_file_contributions),
        ("Mixed File Types", example_4_mixed_file_types),
        ("CURL Examples", example_5_curl_examples)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "="*70)
    print("Running Example 5 (CURL Examples)...")
    print("="*70)
    
    example_5_curl_examples()
    
    print("\n" + "="*70)
    print("To run other examples, modify the main() function")
    print("or call the example functions directly:")
    print("  example_1_basic_docx_contribution()")
    print("  example_2_text_only_contribution()")
    print("  example_3_multiple_file_contributions()")
    print("  example_4_mixed_file_types()")
    print("="*70)


if __name__ == "__main__":
    main()

