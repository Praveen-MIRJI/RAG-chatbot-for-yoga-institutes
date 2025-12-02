# ---------------------------
# Script to Add Institute Metadata to Qdrant
# ---------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Institute metadata
institutes_data = [
    {
        "name": "Sivananda Yoga Vedanta Tapaswini Ashram",
        "code": "YC25120",
        "certification": "Indian Yoga Association: IYA/S-II/010",
        "city": "Nellore",
        "state": "Andhra Pradesh",
        "country": "India",
        "website": "https://sivananda.org.in/gudur/",
        "validity": "N/A"
    },
    {
        "name": "Athayog",
        "code": "YC24108",
        "certification": "YC2400000118",
        "validity": "Jul 2024 - 17 Jul 2027",
        "city": "Bengaluru Urban",
        "state": "Karnataka",
        "country": "India",
        "website": "www.athayogliving.com"
    },
    {
        "name": "Yogmaya Institute Of Yoga Training",
        "code": "YC24114",
        "certification": "YC2400000918",
        "validity": "Jul 2024 - 17 Jul 2027",
        "city": "Jaipur",
        "state": "Rajasthan",
        "country": "India",
        "website": "www.yogmaya.org"
    },
    {
        "name": "Niramaya",
        "code": "YC23099",
        "certification": "1008/21",
        "validity": "May, 2026",
        "city": "Cachar",
        "state": "Assam",
        "country": "India",
        "website": "www.niramayayoga.org"
    },
    {
        "name": "Manappuram Yoga Centre",
        "code": "YC23077",
        "certification": "YAI/IND/KER/24MY2205",
        "validity": "27 Jun 2023 - 26 Jun 2026",
        "city": "Thrissur",
        "state": "Kerala",
        "country": "India",
        "website": "www.manappuramyogacenter.com"
    }
]

def create_embedding(text):
    """Create embedding for text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def add_institute_metadata():
    """Add institute metadata to Qdrant"""
    points = []
    
    for institute in institutes_data:
        # Create a comprehensive text for embedding
        text_for_embedding = f"""
        Institute Name: {institute['name']}
        Code: {institute['code']}
        Location: {institute['city']}, {institute['state']}, {institute['country']}
        Certification: {institute['certification']}
        Validity: {institute.get('validity', 'N/A')}
        Website: {institute['website']}
        
        This is a certified yoga institute located in {institute['city']}, {institute['state']}.
        """
        
        # Generate embedding
        embedding = create_embedding(text_for_embedding)
        
        # Create point with metadata
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "institute_name": institute['name'],
                "code": institute['code'],
                "certification": institute['certification'],
                "validity": institute.get('validity', 'N/A'),
                "city": institute['city'],
                "state": institute['state'],
                "country": institute['country'],
                "website": institute['website'],
                "content": text_for_embedding.strip(),
                "type": "institute_metadata"
            }
        )
        points.append(point)
        print(f"✓ Prepared: {institute['name']}")
    
    # Upload to Qdrant
    try:
        qdrant.upsert(
            collection_name="Institutes",
            points=points
        )
        print(f"\n✅ Successfully added {len(points)} institutes to Qdrant!")
        print("\nInstitutes added:")
        for institute in institutes_data:
            print(f"  • {institute['name']} - {institute['city']}, {institute['state']}")
    except Exception as e:
        print(f"\n❌ Error adding institutes: {e}")

if __name__ == "__main__":
    print("Adding institute metadata to Qdrant...\n")
    add_institute_metadata()
