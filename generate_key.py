import secrets
import base64

def generate_secret_key(length=32):
    """
    Generate a secure secret key.
    Args:
        length: Length of the key in bytes (default: 32 for 256 bits)
    Returns:
        A URL-safe base64-encoded string
    """
    try:
        # Generate random bytes
        random_bytes = secrets.token_bytes(length)
        
        # Convert to URL-safe base64
        secret_key = base64.urlsafe_b64encode(random_bytes).decode('utf-8')
        
        print("\nGenerated Secret Key:")
        print("---------------------")
        print(secret_key)
        print("\nAdd this key to your .env file as:")
        print(f"SECRET_KEY={secret_key}")
        
        return secret_key
    except Exception as e:
        print(f"Error generating secret key: {str(e)}")
        return None

if __name__ == "__main__":
    print("DeepInsight Secret Key Generator")
    print("===============================")
    print("This script generates a secure key for your DeepInsight application.")
    
    # Generate and print the key
    generate_secret_key()