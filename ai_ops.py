import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import re
from blip_caption import BLIPHandler
from image_ocr import SmartOCR

llm = OllamaLLM(model="llama3.2:3b-instruct-q4_K_M")
blip_model = BLIPHandler()
ocr_model = SmartOCR()

rename_prompt = PromptTemplate(
    input_variables=["file_content", "used_names", "mime_type", "additional_info"],
    template="""You are a file naming assistant. Your task is to suggest an appropriate filename based on the content provided, following these strict guidelines:

    1. Analyze the following file content and suggest a descriptive filename:
    {file_content}

    2. Filename requirements:
    - Use only lowercase letters (a-z), numbers (0-9), and hyphens (-)
    - Start the filename with a letter
    - Do not use underscores, spaces, slashes (/), or any special characters
    - Keep the filename concise but descriptive (max 40 characters)
    - Separate words with hyphens
    - Ensure the filename is unique and not in this list: {used_names}
    - Do not include any file extensions (e.g., .txt, .json, etc.)
    - Do not include any metadata (e.g., date, file type) in the filename

    3. Provide a brief but specific description of the file content (15-30 words)

    4. Format your response as a JSON object:
        {{
        "filename": "suggested-filename",
        "description": "Brief specific description of the file content"
        }}

    5. Additional instructions:
    {additional_info}

    Before submitting, double-check that your filename adheres to all requirements, especially avoiding file extensions and metadata.
    Make sure that the words in the file name are not together and are seperated by underscoresand avoid using any extensions in the file name

    Provide only the JSON object in your response, without any additional text or explanation."""
)

categorize_prompt = PromptTemplate(
    input_variables=["file_content", "existing_categories", "cat_strength", "additional_info"],
    template="""You are a file categorization assistant. Your task is to suggest a single appropriate category or folder name for organizing a file based on its content, following these strict guidelines:

    1. Analyze the following file content:
    {file_content}

    2. Categorization strength: {cat_strength}
    - If 1-2: Strongly prefer existing categories
    - If 3-4: Balance between existing and new categories
    - If 5: Prefer creating new, specific categories

    3. Existing categories:
    {existing_categories}

    4. Category name requirements:
    - Use only lowercase letters (a-z), numbers (0-9), and hyphens (-)
    - Start with a letter
    - Keep it short and concise (max 30 characters)
    - Separate words with hyphens
    - Provide only ONE category
    - The category must be directly related to the main topic of the file content
    - The category name must not start with words like 'The', 'Is' etc.
    - Avoid overly broad or generic categories

    5. Additional instructions:
    {additional_info}

    Before submitting, ensure your category is genuinely relevant to the file content and not an unrelated term.
    It is very important to have the category name aligned with the content of the file. Use suitable categories that strictly relate to the file content
    Make sure that the words in the category are not together and are seperated by underscoresand avoid using any extensions in the file name

    Provide only the single category name in your response, without any additional text or explanation."""
)


rename_chain = rename_prompt | llm
categorize_chain = categorize_prompt | llm

def is_markdown_code(text):
    return bool(re.match(r'^\s*```[\w\s]*\n.*\n\s*```\s*$', text, re.DOTALL))

def extract_json_from_text(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

def parse_json(json_string):
    if is_markdown_code(json_string):
        json_string = re.sub(r'^\s*```[\w\s]*\n', '', json_string)
        json_string = re.sub(r'\n\s*```\s*$', '', json_string)
    
    json_string = extract_json_from_text(json_string)
    
    json_string = re.sub(r'(?<!\\)"', '"', json_string)
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', json_string)
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic JSON string: {json_string}")
        fixed_json = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_string)
        fixed_json = json_string + "}"
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            print("Failed to parse JSON even after attempting fixes.")
            return None

def suggest_filename(content, used_names, mime_type, additional_info = ""):
    print(additional_info)
    print(mime_type)
    if mime_type in ['.png', '.jpg', '.jpeg']:
        image_content = ""
        ocr_text = ocr_model.process_image(content)
        if ocr_text:
            image_content += ocr_text
        image_caption = blip_model.generate_caption(content)
        image_content += image_caption
        print(image_content)
        json_response = rename_chain.invoke({"file_content": image_content,"used_names": used_names,"mime_type": mime_type,"additional_info": additional_info})
    else:
        json_response = rename_chain.invoke({"file_content": content,"used_names": used_names,"mime_type": mime_type,"additional_info": additional_info})
    
    result = parse_json(json_response)
    print("JSON : ", result)

    return result

def suggest_category(content, existing_categories, mime_type, cat_strength, additional_info = ""):
    print(additional_info)
    if mime_type in ['.png', '.jpg', '.jpeg']:
        image_content = ""
        ocr_text = ocr_model.process_image(content)
        if ocr_text:
            image_content += ocr_text
        image_caption = blip_model.generate_caption(content)
        image_content += image_caption
        print(image_content)

        cat = categorize_chain.invoke({"file_content": image_content, "existing_categories": existing_categories, "cat_strength": cat_strength, "additional_info" : additional_info})
        cat = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', cat)
        return cat
    else:
        cat = categorize_chain.invoke({"file_content": content, "existing_categories": existing_categories, "cat_strength": cat_strength, "additional_info" : additional_info})
        cat = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', cat)
        return cat