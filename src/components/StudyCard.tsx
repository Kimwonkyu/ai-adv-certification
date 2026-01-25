import { useState, useMemo } from 'react';
import { Question, ConfidenceLevel } from '@/types';
import { cn } from '@/lib/utils';
import { ChevronDown, ChevronUp, Eye, Lightbulb, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { ConfidenceButtons } from './ConfidenceButtons';

interface StudyCardProps {
    question: Question;
    onFeedback: (level: ConfidenceLevel) => void;
}

export function StudyCard({ question, onFeedback }: StudyCardProps) {
    const [showAnswer, setShowAnswer] = useState(false);
    const [showReasoning, setShowReasoning] = useState(false);
    const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

    // Helper to detect and render code blocks
    const renderContent = (text: string) => {
        if (!text) return null;

        // Check if it's a code block (wrapped in ``` or looks like python code)
        const isCode = text.includes('```') ||
            (text.includes('import ') || text.includes('df.') || text.includes('np.'));

        if (isCode) {
            // Clean up triple backticks if present
            const cleanCode = text.replace(/```python|```/g, '').trim();
            return (
                <div className="rounded-xl overflow-hidden my-4 border border-zinc-700 shadow-inner">
                    <SyntaxHighlighter
                        language="python"
                        style={vscDarkPlus}
                        customStyle={{
                            backgroundColor: '#18181b',
                            padding: '1.25rem',
                            fontSize: '15px',
                            fontFamily: 'JetBrains Mono, Menlo, monospace',
                            margin: 0
                        }}
                    >
                        {cleanCode}
                    </SyntaxHighlighter>
                </div>
            );
        }

        return <p className="leading-relaxed whitespace-pre-wrap">{text}</p>;
    };

    const isMcq = question.type === '객관식';

    return (
        <div className="max-w-3xl mx-auto w-full space-y-6">
            <div className="glass-card flex flex-col min-h-[500px] border-zinc-800 shadow-[0_20px_50px_rgba(0,0,0,0.5)] p-8 rounded-3xl bg-[#111111]">
                {/* Header: Accessibility - Large labels */}
                <div className="flex justify-between items-center mb-8 border-b border-zinc-800/50 pb-4">
                    <div className="flex items-center gap-3">
                        <span className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-[10px] font-black text-blue-400 uppercase tracking-[0.2em]">
                            {question.chapter_name}
                        </span>
                        <span className="px-2 py-0.5 bg-zinc-800 rounded text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                            {question.type}
                        </span>
                    </div>
                    <span className={cn(
                        "text-xs font-black uppercase tracking-widest px-2 py-0.5 rounded",
                        question.difficulty === 'easy' ? 'text-emerald-500 bg-emerald-500/5' :
                            question.difficulty === 'medium' ? 'text-blue-500 bg-blue-500/5' : 'text-rose-500 bg-rose-500/5'
                    )}>
                        {question.difficulty}
                    </span>
                </div>

                {/* Question: 18px+ font size for accessibility */}
                <div className="text-[20px] md:text-[22px] font-bold text-white mb-8 tracking-tight">
                    {renderContent(question.question)}
                </div>

                {/* Options vs Mask */}
                <div className="flex-grow">
                    {isMcq && question.options ? (
                        <div className="grid gap-3 mb-6">
                            {question.options.map((opt, idx) => {
                                const isCorrect = opt === question.answer;
                                const isSelected = selectedIdx === idx;

                                return (
                                    <button
                                        key={idx}
                                        disabled={showAnswer}
                                        onClick={() => {
                                            setSelectedIdx(idx);
                                            setShowAnswer(true);
                                        }}
                                        className={cn(
                                            "w-full text-left p-4 rounded-2xl border text-base transition-all duration-300",
                                            !showAnswer
                                                ? "bg-zinc-900/50 border-zinc-800 hover:border-blue-500/50 hover:bg-zinc-800"
                                                : isCorrect
                                                    ? "bg-emerald-500/10 border-emerald-500/50 text-emerald-400 font-bold shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                                                    : isSelected
                                                        ? "bg-rose-500/10 border-rose-500/50 text-rose-400 shadow-[0_0_15px_rgba(244,63,94,0.1)]"
                                                        : "bg-zinc-900/20 border-zinc-900 text-zinc-600"
                                        )}
                                    >
                                        <div className="flex items-center gap-4">
                                            <span className={cn(
                                                "w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-black",
                                                !showAnswer ? "bg-zinc-800 text-zinc-500" :
                                                    isCorrect ? "bg-emerald-500 text-white" :
                                                        isSelected ? "bg-rose-500 text-white" : "bg-zinc-800 text-zinc-700"
                                            )}>
                                                {idx + 1}
                                            </span>
                                            {opt}
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    ) : (
                        // Subjective Mask
                        !showAnswer && (
                            <div
                                className="flex flex-col items-center justify-center h-48 border-2 border-dashed border-zinc-800 rounded-3xl group hover:border-blue-500/50 transition-all cursor-pointer"
                                onClick={() => setShowAnswer(true)}
                            >
                                <div className="w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                                    <Eye className="w-8 h-8 text-blue-500" />
                                </div>
                                <p className="text-zinc-500 font-bold group-hover:text-blue-400 transition-colors uppercase tracking-widest text-sm">정답 확인하기 (Active Recall)</p>
                            </div>
                        )
                    )}

                    {/* Feedback & Reasoning after reveal */}
                    <AnimatePresence>
                        {showAnswer && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="space-y-6 mt-6"
                            >
                                {/* Result Message */}
                                {isMcq ? (
                                    <div className={cn(
                                        "p-6 rounded-2xl border flex items-center gap-4",
                                        selectedIdx !== null && question.options?.[selectedIdx] === question.answer
                                            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                                            : "bg-rose-500/10 border-rose-500/30 text-rose-400"
                                    )}>
                                        <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center shrink-0">
                                            {selectedIdx !== null && question.options?.[selectedIdx] === question.answer ? (
                                                <CheckCircle2 className="w-7 h-7" />
                                            ) : (
                                                <AlertTriangle className="w-7 h-7" />
                                            )}
                                        </div>
                                        <div>
                                            <h4 className="font-black text-xl italic uppercase tracking-tighter">
                                                {selectedIdx !== null && question.options?.[selectedIdx] === question.answer ? "Excellent!" : "Not Quite"}
                                            </h4>
                                            <p className="text-sm opacity-80">
                                                정답은 <span className="font-bold underline decoration-emerald-500/50">{question.answer}</span> 입니다.
                                            </p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="p-6 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                                        <div className="flex items-center gap-2 mb-2">
                                            <CheckCircle2 className="w-5 h-5" />
                                            <span className="text-sm font-black uppercase tracking-tighter">정답 확인</span>
                                        </div>
                                        <div className="text-2xl font-black">{question.answer}</div>
                                    </div>
                                )}

                                {/* Accordion Reasoning */}
                                <button
                                    onClick={() => setShowReasoning(!showReasoning)}
                                    className="w-full flex items-center justify-between p-4 rounded-xl hover:bg-zinc-800 transition-colors border border-zinc-800 text-zinc-300"
                                >
                                    <div className="flex items-center gap-2">
                                        <Lightbulb className="w-5 h-5 text-amber-400" />
                                        <span className="font-bold text-sm">상세 해설 및 학습 힌트</span>
                                    </div>
                                    {showReasoning ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                                </button>

                                {showReasoning && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: 'auto', opacity: 1 }}
                                        className="overflow-hidden space-y-4"
                                    >
                                        <div className="p-5 bg-zinc-800/50 rounded-2xl border border-zinc-700/50">
                                            <h4 className="text-xs font-black text-zinc-500 uppercase mb-3 tracking-widest leading-none">Why?</h4>
                                            <div className="text-base text-zinc-300 leading-relaxed">
                                                {renderContent(question.why)}
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            <div className="p-4 bg-blue-500/5 rounded-xl border border-blue-500/10">
                                                <h4 className="flex items-center gap-2 text-[10px] font-black text-blue-400 uppercase mb-2">
                                                    <Lightbulb className="w-3 h-3" /> Hint
                                                </h4>
                                                <p className="text-sm text-zinc-400">{question.hint}</p>
                                            </div>
                                            {question.trap_points && (
                                                <div className="p-4 bg-rose-500/5 rounded-xl border border-rose-500/10">
                                                    <h4 className="flex items-center gap-2 text-[10px] font-black text-rose-400 uppercase mb-2">
                                                        <AlertTriangle className="w-3 h-3" /> Trap
                                                    </h4>
                                                    <p className="text-sm text-zinc-400">{question.trap_points[0]}</p>
                                                </div>
                                            )}
                                        </div>
                                    </motion.div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Confidence Actions */}
                <div className="mt-12 pt-8 border-t border-zinc-800/50">
                    <p className="text-center text-xs font-black text-zinc-600 uppercase tracking-[0.3em] mb-6">
                        메타인지: 본인의 확신도를 솔직하게 선택하세요
                    </p>
                    <ConfidenceButtons
                        onSelect={onFeedback}
                        disabled={!showAnswer}
                    />
                </div>
            </div>
        </div>
    );
}
