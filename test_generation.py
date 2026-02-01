"""
Quick Test Script for Medical Conversation Generation
======================================================
Run this first to verify your setup works before full generation.

Usage:
    python test_generation.py

This will:
1. Load 3 scenarios from HealthCareMagic
2. Generate 1 conversation with GPT-4 (4 turns)
3. Print the output for inspection
"""

import asyncio
import os
from datetime import datetime

# Check for API keys
def check_api_keys():
    keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY")
    }
    
    print("API Key Status:")
    for name, key in keys.items():
        status = "✓ Set" if key else "✗ Not set"
        print(f"  {name}: {status}")
    
    return keys


async def test_openai():
    """Test OpenAI API connection"""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}],
        max_tokens=20
    )
    return response.choices[0].message.content


async def test_anthropic():
    """Test Anthropic API connection"""
    import anthropic
    
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say 'API test successful' and nothing else."}]
    )
    return response.content[0].text


def test_dataset_loading():
    """Test HuggingFace dataset loading"""
    from datasets import load_dataset
    
    print("\nLoading HealthCareMagic dataset (first 3 rows)...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    
    print(f"Dataset loaded: {len(dataset)} total rows")
    print(f"Columns: {dataset.column_names}")
    
    print("\n" + "="*60)
    print("SAMPLE SCENARIO")
    print("="*60)
    
    sample = dataset[0]
    print(f"\nInstruction: {sample['instruction'][:100]}...")
    print(f"\nPatient Input: {sample['input'][:300]}...")
    print(f"\nDoctor Output: {sample['output'][:300]}...")
    
    return dataset


async def test_mini_conversation():
    """Generate a mini 4-turn conversation to verify the pipeline"""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    
    # Simplified prompts for testing
    physician_system = """You are a physician in a telemedicine consultation. 
    Ask relevant questions and provide medical guidance. Be concise."""
    
    patient_system = """You are a patient describing symptoms. 
    Original complaint: Headache for 3 days, worse in the morning.
    Respond naturally and briefly (1-2 sentences)."""
    
    turns = []
    
    # Turn 0: Patient initial
    patient_initial = "I've had this headache for about 3 days now. It's worse when I wake up."
    turns.append({"role": "patient", "content": patient_initial})
    print(f"\nPatient (Turn 0): {patient_initial}")
    
    # Turn 1: Physician response
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": physician_system},
            {"role": "user", "content": f"Patient says: {patient_initial}\n\nRespond as the physician."}
        ],
        max_tokens=200
    )
    physician_1 = response.choices[0].message.content
    turns.append({"role": "physician", "content": physician_1})
    print(f"\nPhysician (Turn 1): {physician_1}")
    
    # Turn 2: Patient follow-up
    history = f"Patient: {patient_initial}\nPhysician: {physician_1}"
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": patient_system},
            {"role": "user", "content": f"Conversation so far:\n{history}\n\nRespond as the patient."}
        ],
        max_tokens=100
    )
    patient_2 = response.choices[0].message.content
    turns.append({"role": "patient", "content": patient_2})
    print(f"\nPatient (Turn 2): {patient_2}")
    
    # Turn 3: Physician follow-up
    history += f"\nPatient: {patient_2}"
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": physician_system},
            {"role": "user", "content": f"Conversation so far:\n{history}\n\nRespond as the physician."}
        ],
        max_tokens=200
    )
    physician_3 = response.choices[0].message.content
    turns.append({"role": "physician", "content": physician_3})
    print(f"\nPhysician (Turn 3): {physician_3}")
    
    return turns


async def main():
    print("="*60)
    print("MEDICAL CONVERSATION GENERATION - TEST SCRIPT")
    print("="*60)
    
    # 1. Check API keys
    keys = check_api_keys()
    
    # 2. Test dataset loading
    try:
        dataset = test_dataset_loading()
        print("\n✓ Dataset loading successful")
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        return
    
    # 3. Test API connections
    if keys["OpenAI"]:
        print("\nTesting OpenAI API...")
        try:
            result = await test_openai()
            print(f"✓ OpenAI API: {result}")
        except Exception as e:
            print(f"✗ OpenAI API failed: {e}")
    
    if keys["Anthropic"]:
        print("\nTesting Anthropic API...")
        try:
            result = await test_anthropic()
            print(f"✓ Anthropic API: {result}")
        except Exception as e:
            print(f"✗ Anthropic API failed: {e}")
    
    # 4. Test mini conversation generation
    if keys["OpenAI"]:
        print("\n" + "="*60)
        print("GENERATING TEST CONVERSATION (4 turns)")
        print("="*60)
        try:
            turns = await test_mini_conversation()
            print("\n✓ Conversation generation successful!")
            print(f"  Generated {len(turns)} turns")
        except Exception as e:
            print(f"\n✗ Conversation generation failed: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nIf all tests passed, you can run the full generation with:")
    print("  python generate_conversations.py --num_scenarios 100 --turns 8")


if __name__ == "__main__":
    asyncio.run(main())
