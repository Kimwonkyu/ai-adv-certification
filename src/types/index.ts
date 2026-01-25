export type QuestionType = '객관식' | '주관식';
export type ConfidenceLevel = 'again' | 'hard' | 'good' | 'easy';

export interface Question {
    id: string;
    chapter_name: string;
    type: QuestionType;
    question: string;
    options?: string[];
    answer: string;
    why: string;
    hint: string;
    trap_points?: string[];
    difficulty: 'easy' | 'medium' | 'hard';
}

export interface UserProgress {
    completed_ids: string[];
    mistake_ids: string[];
    confidence_map: Record<string, ConfidenceLevel>;
    mastered_ids: string[];
    easy_streak_map: Record<string, number>;
}
