from openai import OpenAI

def gpt_chat_completion(prompt: str, client = None):
    if not client:
        client = OpenAI()
    cnt = 0
    while True:
        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            seed=42,
            temperature=0,
            max_tokens=50,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions well."},
                {"role": "user", "content": prompt},
            ]
            )
            break
        except Exception as e:
            print(e)
            print("Failed to get response")
            if cnt >= 3:
                return None
            cnt += 1
    return response.choices[0].message.content.strip()