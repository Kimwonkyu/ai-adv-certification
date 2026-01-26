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
            <div className="sticky top-16 md:top-20 z-40 bg-[#0a0a0a]/95 backdrop-blur-xl border-b border-zinc-800 pb-3 mb-6 pt-4">
                <div className="flex items-center justify-between gap-2 md:gap-4 mb-3">
                    <button
                        onClick={onBack}
                        className="p-1.5 md:p-2 hover:bg-zinc-800 rounded-full transition-colors text-zinc-400 hover:text-white group shrink-0"
                        title="대시보드로 돌아가기"
                    >
                        <ArrowLeft className="w-5 h-5 md:w-6 md:h-6 group-hover:-translate-x-1 transition-transform" />
                    </button>
                    <div className="flex-1 min-w-0">
                        <h1 className="text-lg md:text-2xl font-black text-white uppercase italic tracking-tighter truncate flex items-center gap-2">
                            DB <span className="text-blue-500 not-italic">EXPLORER</span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                        <button
                            onClick={expandAll}
                            className="p-1.5 md:p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
                            title="모두 펼치기"
                        >
                            <Maximize2 className="w-4 h-4 md:w-5 md:h-5" />
                        </button>
                        <button
                            onClick={collapseAll}
                            className="p-1.5 md:p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
                            title="모두 접기"
                        >
                            <Minimize2 className="w-4 h-4 md:w-5 md:h-5" />
                        </button>
                    </div>
                </div>
                <div className="flex items-center justify-between text-[10px] md:text-xs font-bold text-zinc-500 uppercase tracking-widest px-1">
                    <span>Total {questions.length} Items</span>
                    <span>{Object.keys(expandedChapters).filter(k => expandedChapters[k]).length} / {chapterNames.length} Active</span>
                </div>
            </div>

            <div className="space-y-4 md:space-y-6">
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
                                className="w-full px-4 md:px-6 py-4 md:py-5 flex items-center justify-between transition-colors cursor-pointer"
                            >
                                <div className="flex items-center gap-3 md:gap-4 overflow-hidden">
                                    <div className={cn(
                                        "w-10 h-10 md:w-12 md:h-12 rounded-xl flex items-center justify-center border transition-all duration-300 shrink-0",
                                        isExpanded
                                            ? "bg-blue-600 text-white border-blue-500 shadow-lg shadow-blue-900/20"
                                            : "bg-zinc-800/50 text-zinc-500 border-zinc-700"
                                    )}>
                                        <FileCode className="w-5 h-5 md:w-6 md:h-6" />
                                    </div>
                                    <div className="text-left min-w-0">
                                        <h3 className={cn(
                                            "text-base md:text-lg font-black uppercase italic transition-colors truncate",
                                            isExpanded ? "text-white" : "text-zinc-400"
                                        )}>{chapter}</h3>
                                        <p className="text-[10px] text-zinc-500 font-bold tracking-widest uppercase mt-0.5 truncate">
                                            {items.length} Qs • {items.filter(i => i.type === '객관식').length} MCQs
                                        </p>
                                    </div>
                                </div>
                                {isExpanded
                                    ? <ChevronUp className="w-5 h-5 text-blue-500 shrink-0" />
                                    : <ChevronDown className="w-5 h-5 text-zinc-600 shrink-0" />
                                }
                            </button>

                            {isExpanded && (
                                <div className="px-4 md:px-6 pb-6 space-y-4 md:space-y-6 animate-in slide-in-from-top-2 duration-300 border-t border-zinc-800/50 pt-4 md:pt-6">
                                    {items.map((q, idx) => (
                                        <div key={q.id} className="relative">
                                            {/* Timeline Connector for Desktop */}
                                            <div className="hidden md:block absolute left-[-24px] top-0 bottom-0 w-px bg-zinc-800 last:bottom-auto last:h-full"></div>

                                            <div className="p-4 md:p-5 bg-black/40 rounded-2xl border border-zinc-800 hover:border-zinc-700 transition-colors group">
                                                {/* Header & Badges */}
                                                <div className="flex flex-col gap-3 mb-4">
                                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                                        <div className="flex items-center gap-2">
                                                            <span className="px-2 py-1 rounded-md bg-blue-500/10 text-blue-400 text-[10px] font-black uppercase tracking-widest border border-blue-500/20">
                                                                Q{idx + 1}
                                                            </span>
                                                            <span className="text-[10px] font-bold text-zinc-600 uppercase tracking-wider">
                                                                ID: {q.id}
                                                            </span>
                                                        </div>
                                                        <div className={cn(
                                                            "px-2 py-1 rounded-lg border text-[10px] font-black uppercase tracking-widest",
                                                            q.difficulty === 'hard' ? "bg-rose-500/10 border-rose-500/20 text-rose-500" :
                                                                q.difficulty === 'medium' ? "bg-amber-500/10 border-amber-500/20 text-amber-500" :
                                                                    "bg-emerald-500/10 border-emerald-500/20 text-emerald-500"
                                                        )}>
                                                            {q.difficulty}
                                                        </div>
                                                    </div>
                                                    <h4 className="text-sm md:text-base font-bold text-zinc-200 leading-relaxed whitespace-pre-wrap word-break-keep-all">
                                                        {q.question}
                                                    </h4>
                                                </div>

                                                {/* Answer Section */}
                                                <div className="mb-4 md:mb-6 bg-zinc-900/30 rounded-xl p-3 md:p-4 border border-zinc-800/50">
                                                    <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest mb-2 md:mb-3 block flex items-center gap-2">
                                                        <CheckCircle2 className="w-3 h-3" /> Answer
                                                    </span>
                                                    {q.options && q.options.length > 0 ? (
                                                        <div className="grid grid-cols-1 gap-1.5 md:gap-2">
                                                            {q.options.map((opt, i) => (
                                                                <div key={i} className={cn(
                                                                    "px-3 md:px-4 py-2.5 md:py-3 rounded-lg text-xs md:text-sm font-medium border transition-all break-words",
                                                                    opt === q.answer
                                                                        ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400 font-bold shadow-sm"
                                                                        : "bg-black/20 border-transparent text-zinc-600"
                                                                )}>
                                                                    <div className="flex items-start gap-2">
                                                                        <span className="opacity-50 mt-0.5">{i + 1}.</span>
                                                                        <span className="flex-1">{opt}</span>
                                                                        {opt === q.answer && <span className="ml-1 text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded uppercase font-bold shrink-0">Correct</span>}
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    ) : (
                                                        <div className="bg-black/40 border border-emerald-500/20 p-3 md:p-4 rounded-lg font-mono text-xs md:text-sm text-emerald-400 overflow-x-auto">
                                                            {q.answer}
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Explanation Section */}
                                                <div>
                                                    <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest mb-1.5 md:mb-2 block">
                                                        Explanation
                                                    </span>
                                                    <p className="text-xs md:text-sm text-zinc-400 leading-relaxed bg-zinc-900/20 p-3 md:p-4 rounded-xl border border-zinc-800/30 break-words">
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
