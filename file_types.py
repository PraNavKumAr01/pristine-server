from mimetypes import guess_type

image_exts = [".png", ".jpg", ".jpeg"]
doc_exts = [".doc", ".docx"]
txt_exts = [".txt"]
pdf_exts = [".pdf"]
excel_exts = [".xlsx"]
csv_exts = [".csv"]

mime_type_map = {
    'text/plain': '.txt',
    'image/jpeg': '.jpeg',
    'image/png': '.png',
    'image/gif': '.gif',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'text/csv': '.csv',
    'text/javascript': '.js',
    'text/html': '.html',
    'text/css': '.css',
    'text/x-python': '.py',
    'application/json': '.json',
    'application/xml': '.xml',
    'application/x-sh': '.sh',
    'application/x-httpd-php': '.php',
    'application/sql': '.sql',
    'text/x-java-source': '.java',
    'application/x-ruby': '.rb',
    'text/x-c': '.c',
    'text/x-c++': '.cpp',
    'application/x-perl': '.pl',
    'application/x-tcl': '.tcl',
    'text/x-lisp': '.lisp',
    'application/x-scheme': '.scm',
    'text/x-r': '.r'
}

def get_file_extension(file_path):
    mime_type, _ = guess_type(file_path)
    return mime_type_map.get(mime_type)