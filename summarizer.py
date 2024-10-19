from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
import re
import json

llm_text = Ollama(model="qwen2.5:1.5b-instruct")
llm_image = Ollama(model = "moondream:v2")

main_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
    Based on the following file content, respond with a descriptive and concise summary of the content. 
    The summary should be short and clear and should include all the important themes and aspects of that content.
    The file content is 
    {content}
    Your response should be a JSON with a key 'summary' and it should contain only the summary
    Respond with the JSON and nothing else
    """,
)

exe_chain = main_prompt | llm_text

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

def summarize_text(content):

    res = exe_chain.invoke({"content": content})
    res = parse_json(res)

    return res["summary"]

def summarize_image(content):
    llm_with_image_context = llm_image.bind(images=[content])
    prompt="""
    Based on the following file content, respond with a descriptive and concise summary of the content. 
    The summary should be short and clear and should include all the important themes and aspects of that content. 
    Your response should be a JSON with a key 'summary' and it should contain only the summary. 
    Respond with the JSON and nothing else
    """
    result = llm_with_image_context.invoke(prompt)
    result = parse_json(result)

    return result["summary"]