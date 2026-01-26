import { Question } from '@/types';
import { ChevronDown, ChevronUp, FileCode, CheckCircle2, ArrowLeft, Maximize2, Minimize2 } from 'lucide-react';
import { useState, useMemo } from 'react';
import { cn } from '@/lib/utils';

interface HiddenQuestionListProps {
    questions: Question[];
    onBack: () => void;
}

export function HiddenQuestionList({ questions, onBack }: HiddenQuestionListProps) {
    const groupedQuestions = useMemo(() => {
        const groups: Record<string, Question[]> = {};
        questions.forEach(q => {
            if (!groups[q.chapter_name]) {
                groups[q.chapter_name] = [];
            }
            groups[q.chapter_name].push(q);
        });
        return groups;
    }, [questions]);

    const chapterNames = Object.keys(groupedQuestions);
    const [expandedChapters, setExpandedChapters] = useState<Record<string, boolean>>({});

    const toggleChapter = (chapter: string) => {
        setExpandedChapters(prev => ({
            ...prev,
            [chapter]: !prev[chapter]
        }));
    };

    const expandAll = () => {
        const all: Record<string, boolean> = {};
        chapterNames.forEach(c => all[c] = true);
        setExpandedChapters(all);
    };

    const collapseAll = () => {
        setExpandedChapters({});
    };

    return (
        <div className="pb-20 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Sticky Header */}
            <div className="sticky top-20 z-40 bg-[#0a0a0a]/95 backdrop-blur-xl border-b border-zinc-800 pb-4 mb-8 pt-2 -mx-2 px-2">
                <div className="flex items-center justify-between gap-4 mb-4">
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-zinc-800 rounded-full transition-colors text-zinc-400 hover:text-white group"
                        title="대시보드로 돌아가기"
                    >
                        <ArrowLeft className="w-6 h-6 group-hover:-translate-x-1 transition-transform" />
                    </button>
                    <div className="flex-1">
                        <h1 className="text-2xl font-black text-white uppercase italic tracking-tighter flex items-center gap-2">
                            DATABASE <span className="text-blue-500 not-italic">EXPLORER</span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={expandAll}
                            className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
                            title="모두 펼치기"
                        >
                            <Maximize2 className="w-5 h-5" />
                        </button>
                        <button
                            onClick={collapseAll}
                            className="p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
                            title="모두 접기"
                        >
                            <Minimize2 className="w-5 h-5" />
                        </button>
                    </div>
                </div>
                <div className="flex items-center justify-between text-xs font-bold text-zinc-500 uppercase tracking-widest px-2">
                    <span>Total {questions.length} Items</span>
                    <span>{Object.keys(expandedChapters).filter(k => expandedChapters[k]).length} / {chapterNames.length} Chapters Active</span>
                </div>
            </div>

            <div className="space-y-6">
                {chapterNames.map((chapter) => {
                    const items = groupedQuestions[chapter];
                    const isExpanded = expandedChapters[chapter];

                    return (
                        <div key={chapter} className={cn(
                            "glass-card border-zinc-800 bg-zinc-900/40 overflow-hidden transition-all duration-500",
                            isExpanded ? "ring-1 ring-blue-500/30" : "hover:bg-zinc-900/60"
                        )}>
                            <button
                                onClick={() => toggleChapter(chapter)}
                                className="w-full px-6 py-5 flex items-center justify-between transition-colors cursor-pointer"
                            >
                                <div className="flex items-center gap-4">
                                    <div className={cn(
                                        "w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-300",
                                        isExpanded
                                            ? "bg-blue-600 text-white border-blue-500 shadow-lg shadow-blue-900/20"
                                            : "bg-zinc-800/50 text-zinc-500 border-zinc-700"
                                    )}>
                                        <FileCode className="w-6 h-6" />
                                    </div>
                                    <div className="text-left">
                                        <h3 className={cn(
                                            "text-lg font-black uppercase italic transition-colors",
                                            isExpanded ? "text-white" : "text-zinc-400"
                                        )}>{chapter}</h3>
                                        <p className="text-[10px] text-zinc-500 font-bold tracking-widest uppercase mt-1">
                                            {items.length} Questions • {items.filter(i => i.type === '객관식').length} Multiple Choice
                                        </p>
                                    </div>
                                </div>
                                {isExpanded
                                    ? <ChevronUp className="w-5 h-5 text-blue-500" />
                                    : <ChevronDown className="w-5 h-5 text-zinc-600" />
                                }
                            </button>

                            {isExpanded && (
                                <div className="px-6 pb-6 space-y-6 animate-in slide-in-from-top-2 duration-300 border-t border-zinc-800/50 pt-6">
                                    {items.map((q, idx) => (
                                        <div key={q.id} className="relative pl-4 md:pl-0">
                                            {/* Timeline Connector for Desktop */}
                                            <div className="hidden md:block absolute left-[-24px] top-0 bottom-0 w-px bg-zinc-800 last:bottom-auto last:h-full"></div>

                                            <div className="p-5 bg-black/40 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-colors group">
                                                {/* Header & Badges */}
                                                <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-6">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-3 mb-3">
                                                            <span className="px-2.5 py-1 rounded-md bg-blue-500/10 text-blue-400 text-[10px] font-black uppercase tracking-widest border border-blue-500/20">
                                                                Question {idx + 1}
                                                            </span>
                                                            <span className="text-[10px] font-bold text-zinc-600 uppercase tracking-wider">
                                                                ID: {q.id}
                                                            </span>
                                                        </div>
                                                        <h4 className="text-base font-bold text-zinc-200 leading-relaxed whitespace-pre-wrap">{q.question}</h4>
                                                    </div>
                                                    <div className={cn(
                                                        "shrink-0 px-3 py-1.5 rounded-lg border text-[10px] font-black uppercase tracking-widest self-start",
                                                        q.difficulty === 'hard' ? "bg-rose-500/10 border-rose-500/20 text-rose-500" :
                                                            q.difficulty === 'medium' ? "bg-amber-500/10 border-amber-500/20 text-amber-500" :
                                                                "bg-emerald-500/10 border-emerald-500/20 text-emerald-500"
                                                    )}>
                                                        {q.difficulty} Level
                                                    </div>
                                                </div>

                                                {/* Answer Section */}
                                                <div className="mb-6 bg-zinc-900/30 rounded-xl p-4 border border-zinc-800/50">
                                                    <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest mb-3 block flex items-center gap-2">
                                                        <CheckCircle2 className="w-3 h-3" /> Answer
                                                    </span>
                                                    {q.options && q.options.length > 0 ? (
                                                        <div className="grid grid-cols-1 gap-2">
                                                            {q.options.map((opt, i) => (
                                                                <div key={i} className={cn(
                                                                    "px-4 py-3 rounded-lg text-sm font-medium border transition-all",
                                                                    opt === q.answer
                                                                        ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400 font-bold shadow-sm"
                                                                        : "bg-black/20 border-transparent text-zinc-600"
                                                                )}>
                                                                    <span className="mr-3 opacity-50">{i + 1}.</span>
                                                                    {opt}
                                                                    {opt === q.answer && <span className="ml-2 text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded uppercase font-bold">Correct</span>}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    ) : (
                                                        <div className="bg-black/40 border border-emerald-500/20 p-4 rounded-lg font-mono text-sm text-emerald-400 overflow-x-auto">
                                                            {q.answer}
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Explanation Section */}
                                                <div>
                                                    <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-2 block">
                                                        Explanation
                                                    </span>
                                                    <p className="text-sm text-zinc-400 leading-relaxed bg-zinc-900/20 p-4 rounded-xl border border-zinc-800/30">
                                                        {q.why}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
