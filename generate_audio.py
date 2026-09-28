#!/usr/bin/env python
"""Generate audio MP3 files from a stored conversation."""

import sqlite3
from gtts import gTTS
from pathlib import Path

def generate_audio_from_conversation(conv_id=23):
    """Convert a conversation to MP3 files."""
    
    with sqlite3.connect('voice_agent.db') as conn:
        cursor = conn.cursor()
        
        # Get conversation
        cursor.execute('SELECT id, prompt_version, persona FROM conversations WHERE id = ?', (conv_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"Conversation {conv_id} not found!")
            return
        
        conv_id, version, persona = result
        
        print(f'\n🎙️  Converting Conversation #{conv_id} to Audio')
        print(f'   Version: {version} | Persona: {persona}')
        print('-' * 70)
        
        # Get turns
        cursor.execute(
            'SELECT turn_idx, speaker, text FROM turns WHERE conversation_id = ? ORDER BY turn_idx',
            (conv_id,)
        )
        turns = cursor.fetchall()
        
        output_dir = Path('outputs/audio')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for turn_idx, speaker, text in turns:
            if not text or not text.strip():
                print(f'[Turn {turn_idx}] {speaker}: [Empty]')
                continue
            
            # Clean up text for TTS
            text = text.strip()[:300]
            
            # Detect language
            has_hindi = any(c in text for c in ['ा', 'ि', 'ु', 'े', 'ो', 'ै', 'ण', 'ड', 'ठ', 'थ', 'क', 'ग'])
            hindi_words = ['main', 'aap', 'hoon', 'hai', 'kya', 'hain', 'kaun', 'nahi', 'bilkul', 'theek']
            has_hindi_word = any(word in text.lower() for word in hindi_words)
            
            lang = 'hi' if (has_hindi or has_hindi_word) else 'en'
            
            try:
                # Different accent for agent vs customer
                tld = 'co.in' if speaker == 'agent' else 'com'
                
                tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
                filename = output_dir / f'conv{conv_id}_turn{turn_idx:02d}_{speaker}.mp3'
                tts.save(str(filename))
                
                speaker_name = '🤖 Agent (Riya)' if speaker == 'agent' else f'👤 Customer ({persona})'
                print(f'\n[Turn {turn_idx}] {speaker_name}')
                print(f'   Text: {text}')
                print(f'   Audio: {filename.name} ✓')
                
            except Exception as e:
                print(f'[Turn {turn_idx}] Error: {e}')

        print('\n' + '='*70)
        print(f'✅ Audio files saved to: outputs/audio/')
        print(f'   These are real MP3 files you can listen to!')
        print(f'   Location: {output_dir.absolute()}')
        print('='*70 + '\n')
        
        # List the files created
        print('📁 Files created:')
        for mp3_file in sorted(output_dir.glob(f'conv{conv_id}_*.mp3')):
            size_kb = mp3_file.stat().st_size / 1024
            print(f'   ✓ {mp3_file.name} ({size_kb:.1f} KB)')


if __name__ == '__main__':
    generate_audio_from_conversation(conv_id=23)
