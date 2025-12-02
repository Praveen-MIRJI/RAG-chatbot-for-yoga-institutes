# Deployment Guide for Streamlit Cloud

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Your API keys ready

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure you have all these files:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Excludes secrets from git
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `README.md` - Project documentation

### 2. Initialize Git and Push to GitHub

```bash
# Initialize git repository
git init

# Add all files (secrets.toml will be ignored)
git add .

# Commit
git commit -m "Initial commit: Yoga Institute AI Assistant"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Click **"New app"**

3. Connect your GitHub account if not already connected

4. Select:
   - **Repository**: Your repository name
   - **Branch**: main
   - **Main file path**: app.py

5. Click **"Advanced settings"**

6. In the **Secrets** section, paste your API keys in TOML format:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
QDRANT_URL = "your-qdrant-url-here"
QDRANT_API_KEY = "your-qdrant-api-key-here"
```

7. Click **"Deploy"**

### 4. Wait for Deployment

- Streamlit will install dependencies and start your app
- This usually takes 2-3 minutes
- You'll get a public URL like: `https://your-app-name.streamlit.app`

### 5. Test Your Deployment

1. Visit your app URL
2. Try asking: "Hello"
3. Try asking: "What institutes are available?"
4. Test a specific query about an institute

## Updating Your App

After making changes locally:

```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy your app!

## Troubleshooting

### App won't start
- Check the logs in Streamlit Cloud dashboard
- Verify all secrets are correctly set
- Ensure requirements.txt has correct package versions

### API errors
- Verify your OpenAI API key is valid and has credits
- Check Qdrant URL and API key are correct
- Test API keys locally first

### Secrets not working
- Make sure secrets are in TOML format
- No quotes around the key names
- Use quotes around the values
- Check for typos in secret names

## Security Notes

- ✅ Never commit `.streamlit/secrets.toml` to git
- ✅ Always use Streamlit secrets for API keys
- ✅ Rotate API keys if accidentally exposed
- ✅ Monitor your OpenAI usage dashboard

## Cost Management

- Monitor token usage in the app sidebar
- Set spending limits in OpenAI dashboard
- Consider adding rate limiting for production use

## Support

For issues:
- Streamlit docs: [docs.streamlit.io](https://docs.streamlit.io)
- Streamlit community: [discuss.streamlit.io](https://discuss.streamlit.io)
