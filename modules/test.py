import asyncio
import random

# Define the shared array
shared_array = []

async def populate_array():
    while True:
        # Generate a random number and append to the array
        num = random.randint(1, 100)
        shared_array.append(num)
        print(f"Added: {num}")
        # Wait for a random amount of time before adding the next number
        await asyncio.sleep(random.uniform(0.1, 1.0))

async def print_latest_number():
    while True:
        if shared_array:
            # Print the latest number in the array
            print(f"Latest: {shared_array[-1]}")
        # Wait for a short amount of time before checking again
        await asyncio.sleep(0.5)

async def main():
    # Run both functions concurrently
    await asyncio.gather(populate_array(), print_latest_number())

# Entry point to the script
if __name__ == "__main__":
    asyncio.run(main())
