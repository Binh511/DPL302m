import os
import csv
import re
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import urlparse, parse_qs
import argparse
from datetime import datetime

class YouTubeCommentScraper:
    def __init__(self, api_key):
        """
        Initialize the YouTube Comment Scraper
        
        Args:
            api_key (str): YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def extract_video_id(self, youtube_url):
        """
        Extract video ID from YouTube URL
        
        Args:
            youtube_url (str): YouTube video URL
            
        Returns:
            str: Video ID or None if invalid URL
        """
        # Regular expression patterns for different YouTube URL formats
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:v\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        return None
    
    def get_video_info(self, video_id):
        """
        Get basic video information
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            dict: Video information
        """
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                video = response['items'][0]
                return {
                    'title': video['snippet']['title'],
                    'channel': video['snippet']['channelTitle'],
                    'published_at': video['snippet']['publishedAt'],
                    'view_count': video['statistics'].get('viewCount', 0),
                    'like_count': video['statistics'].get('likeCount', 0),
                    'comment_count': video['statistics'].get('commentCount', 0)
                }
            else:
                return None
        except HttpError as e:
            print(f"An error occurred while fetching video info: {e}")
            return None
    
    def get_comments(self, video_id, max_results=None):
        """
        Fetch all comments from a YouTube video
        
        Args:
            video_id (str): YouTube video ID
            max_results (int): Maximum number of comments to fetch (None for all)
            
        Returns:
            list: List of comment dictionaries
        """
        comments = []
        next_page_token = None
        
        try:
            while True:
                # Request comments
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,  # Maximum allowed per request
                    pageToken=next_page_token,
                    order='relevance'  # Can be 'time' or 'relevance'
                )
                
                response = request.execute()
                
                # Process comments
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    comment_data = {
                        'comment_id': item['snippet']['topLevelComment']['id'],
                        'author': comment['authorDisplayName'],
                        'author_channel_id': comment.get('authorChannelId', {}).get('value', ''),
                        'text': comment['textDisplay'],
                        'text_original': comment['textOriginal'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment['updatedAt'],
                        'reply_count': item['snippet']['totalReplyCount']
                    }
                    
                    comments.append(comment_data)
                    
                    # Get replies if they exist
                    if item['snippet']['totalReplyCount'] > 0:
                        replies = self.get_comment_replies(item['snippet']['topLevelComment']['id'])
                        comments.extend(replies)
                
                # Check if we've reached the max results limit
                if max_results and len(comments) >= max_results:
                    comments = comments[:max_results]
                    break
                
                # Check for next page
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                
                print(f"Fetched {len(comments)} comments so far...")
        
        except HttpError as e:
            print(f"An error occurred while fetching comments: {e}")
            if "commentsDisabled" in str(e):
                print("Comments are disabled for this video.")
            elif "quotaExceeded" in str(e):
                print("API quota exceeded. Please try again later.")
        
        return comments
    
    def get_comment_replies(self, comment_id):
        """
        Fetch replies to a specific comment
        
        Args:
            comment_id (str): Parent comment ID
            
        Returns:
            list: List of reply dictionaries
        """
        replies = []
        next_page_token = None
        
        try:
            while True:
                request = self.youtube.comments().list(
                    part='snippet',
                    parentId=comment_id,
                    maxResults=100,
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response['items']:
                    reply = item['snippet']
                    
                    reply_data = {
                        'comment_id': item['id'],
                        'author': reply['authorDisplayName'],
                        'author_channel_id': reply.get('authorChannelId', {}).get('value', ''),
                        'text': reply['textDisplay'],
                        'text_original': reply['textOriginal'],
                        'like_count': reply['likeCount'],
                        'published_at': reply['publishedAt'],
                        'updated_at': reply['updatedAt'],
                        'reply_count': 0,  # Replies don't have replies
                        'parent_id': comment_id
                    }
                    
                    replies.append(reply_data)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
        
        except HttpError as e:
            print(f"An error occurred while fetching replies: {e}")
        
        return replies
    
    def save_to_csv(self, comments, video_info, filename=None):
        """
        Save comments to CSV file
        
        Args:
            comments (list): List of comment dictionaries
            video_info (dict): Video information
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            # Create filename based on video title and timestamp
            safe_title = re.sub(r'[^\w\s-]', '', video_info['title'])
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_comments_{safe_title}_{timestamp}.csv"
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(comments)
        
        # Add video information to the DataFrame
        for key, value in video_info.items():
            df[f'video_{key}'] = value
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"Comments saved to: {filename}")
        print(f"Total comments saved: {len(comments)}")
        
        return filename
    
    def scrape_comments(self, youtube_url, output_file=None, max_results=None):
        """
        Main method to scrape comments from a YouTube video
        
        Args:
            youtube_url (str): YouTube video URL
            output_file (str): Output CSV filename (optional)
            max_results (int): Maximum number of comments to fetch (optional)
            
        Returns:
            tuple: (comments_list, video_info, output_filename)
        """
        print(f"Starting to scrape comments from: {youtube_url}")
        
        # Extract video ID
        video_id = self.extract_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL. Please provide a valid YouTube video URL.")
        
        print(f"Video ID: {video_id}")
        
        # Get video information
        video_info = self.get_video_info(video_id)
        if not video_info:
            raise ValueError("Could not fetch video information. The video might be private or deleted.")
        
        print(f"Video Title: {video_info['title']}")
        print(f"Channel: {video_info['channel']}")
        print(f"Expected Comments: {video_info['comment_count']}")
        
        # Fetch comments
        print("Fetching comments...")
        comments = self.get_comments(video_id, max_results)
        
        if not comments:
            print("No comments found or comments are disabled for this video.")
            return [], video_info, None
        
        # Save to CSV
        output_filename = self.save_to_csv(comments, video_info, output_file)
        
        return comments, video_info, output_filename


def main():
    """
    Command line interface for the YouTube Comment Scraper
    """
    parser = argparse.ArgumentParser(description='Scrape comments from YouTube videos')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--api-key', required=True, help='YouTube Data API v3 key')
    parser.add_argument('--output', '-o', help='Output CSV filename')
    parser.add_argument('--max-results', '-m', type=int, help='Maximum number of comments to fetch')
    
    args = parser.parse_args()
    
    try:
        # Initialize scraper
        scraper = YouTubeCommentScraper(args.api_key)
        
        # Scrape comments
        comments, video_info, output_file = scraper.scrape_comments(
            args.url, 
            args.output, 
            args.max_results
        )
        
        print("\n" + "="*50)
        print("SCRAPING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Video: {video_info['title']}")
        print(f"Comments fetched: {len(comments)}")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage without command line arguments
    # You can uncomment and modify this section for direct usage
    
    # API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"
    # VIDEO_URL = "https://www.youtube.com/watch?v=VIDEO_ID"
    # 
    # scraper = YouTubeCommentScraper(API_KEY)
    # comments, video_info, output_file = scraper.scrape_comments(VIDEO_URL)
    
    main()