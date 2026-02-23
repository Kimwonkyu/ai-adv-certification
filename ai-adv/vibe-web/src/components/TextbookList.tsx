import { BookOpen, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Textbook {
    id: number;
    title: string;
    description: string;
    color: string;
}

export const TEXTBOOKS: Textbook[] = [
    { id: 1, title: 'Python 기초', description: '파이썬의 기초 문법과 핵심 개념을 학습합니다.', color: 'text-blue-500 bg-blue-500/10 border-blue-500/20' },
    { id: 2, title: '데이터 분석', description: 'Pandas와 NumPy를 활용한 데이터 분석 입문.', color: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20' },
    { id: 3, title: 'LLM 기본', description: '대거대 언어 모델의 원리와 기초 이론.', color: 'text-purple-500 bg-purple-500/10 border-purple-500/20' },
    { id: 4, title: '프롬프트 엔지니어링', description: '효과적인 프롬프트 작성 기법과 전략.', color: 'text-amber-500 bg-amber-500/10 border-amber-500/20' },
    { id: 5, title: 'RAG Agent', description: '검색 증강 생성(RAG)과 에이전트 구축 실습.', color: 'text-rose-500 bg-rose-500/10 border-rose-500/20' },
    { id: 6, title: 'Fine Tuning', description: '모델 성능 향상을 위한 파인 튜닝 기법.', color: 'text-cyan-500 bg-cyan-500/10 border-cyan-500/20' },
];

interface TextbookListProps {
    onSelect: (id: number) => void;
}

export function TextbookList({ onSelect }: TextbookListProps) {
    return (
        <div className="animate-in fade-in duration-700 space-y-8">
            <div className="flex items-center gap-3 mb-8">
                <div className="w-12 h-12 bg-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
                    <BookOpen className="w-6 h-6 text-white" />
                </div>
                <div>
                    <h2 className="text-3xl font-black text-white uppercase italic tracking-tighter">AI Certification <span className="text-indigo-500">Textbook</span></h2>
                    <p className="text-zinc-400 font-medium">단원별 핵심 이론 교재를 학습하세요.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {TEXTBOOKS.map((book) => (
                    <button
                        key={book.id}
                        onClick={() => onSelect(book.id)}
                        className={cn(
                            "group text-left p-6 rounded-2xl border transition-all duration-300 hover:scale-[1.02]",
                            "bg-zinc-900/50 hover:bg-zinc-900 border-zinc-800",
                            "hover:shadow-xl hover:border-zinc-700"
                        )}
                    >
                        <div className={cn("inline-flex p-3 rounded-xl mb-4", book.color)}>
                            <BookOpen className="w-6 h-6" />
                        </div>
                        <h3 className="text-xl font-bold text-white mb-2 group-hover:text-indigo-400 transition-colors">
                            {book.id}. {book.title}
                        </h3>
                        <p className="text-sm text-zinc-400 leading-relaxed mb-6">
                            {book.description}
                        </p>
                        <div className="flex items-center text-sm font-bold text-zinc-500 group-hover:text-white transition-colors">
                            학습하기 <ArrowRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
}
