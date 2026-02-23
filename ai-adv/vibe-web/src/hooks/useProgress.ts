import { useState, useEffect } from 'react';
import { UserProgress, ConfidenceLevel } from '@/types';

const STORAGE_KEY = 'vibe_study_progress_v2';

export function useProgress() {
    const [progress, setProgress] = useState<UserProgress>({
        completed_ids: [],
        mistake_ids: [],
        confidence_map: {},
        mastered_ids: [],
        easy_streak_map: {},
    });

    useEffect(() => {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                setProgress(JSON.parse(saved));
            } catch (e) {
                console.error('Failed to parse progress', e);
            }
        }
    }, []);

    const saveProgress = (newProgress: UserProgress) => {
        setProgress(newProgress);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newProgress));
    };

    const updateConfidence = (id: string, confidence: ConfidenceLevel) => {
        const newProgress = { ...progress };

        // 1. Mark as completed
        if (!newProgress.completed_ids.includes(id)) {
            newProgress.completed_ids.push(id);
        }

        // 2. Update confidence map
        newProgress.confidence_map[id] = confidence;

        // 3. Handle mistake (Again)
        if (confidence === 'again') {
            if (!newProgress.mistake_ids.includes(id)) {
                newProgress.mistake_ids.push(id);
            }
            newProgress.easy_streak_map[id] = 0;
        } else {
            newProgress.mistake_ids = newProgress.mistake_ids.filter(mid => mid !== id);
        }

        // 4. Handle mastery logic (2 consecutive Easy)
        if (confidence === 'easy') {
            const currentStreak = (newProgress.easy_streak_map[id] || 0) + 1;
            newProgress.easy_streak_map[id] = currentStreak;

            if (currentStreak >= 2 && !newProgress.mastered_ids.includes(id)) {
                newProgress.mastered_ids.push(id);
            }
        } else if (confidence === 'good' || confidence === 'hard') {
            newProgress.easy_streak_map[id] = 0;
        }

        saveProgress(newProgress);
    };

    const resetProgress = () => {
        const emptyProgress: UserProgress = {
            completed_ids: [],
            mistake_ids: [],
            confidence_map: {},
            mastered_ids: [],
            easy_streak_map: {},
        };
        saveProgress(emptyProgress);
        localStorage.removeItem(STORAGE_KEY);
    };

    return { progress, updateConfidence, resetProgress };
}
