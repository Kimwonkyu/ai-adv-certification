import { useEffect, useState } from 'react';
import { ArrowLeft, BookOpen, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Textbook } from './TextbookList';

interface TextbookViewerProps {
    chapterId: number;
    onBack: () => void;
}

export function TextbookViewer({ chapterId, onBack }: TextbookViewerProps) {
    const [content, setContent] = useState<string>('');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`/textbooks/${chapterId}.md`)
            .then(res => res.text())
            .then(text => {
                setContent(text);
                setLoading(false);
            })
            .catch(err => {
                console.error('Failed to load textbook:', err);
                setLoading(false);
            });
    }, [chapterId]);

    const parseLine = (line: string, index: number) => {
        // Headers
        if (line.startsWith('# ')) return <h1 key={index} className="text-3xl font-black text-white mt-8 mb-6 pb-4 border-b border-zinc-800">{parseBold(line.slice(2))}</h1>;
        if (line.startsWith('## ')) return <h2 key={index} className="text-2xl font-bold text-white mt-8 mb-4">{parseBold(line.slice(3))}</h2>;
        if (line.startsWith('### ')) return <h3 key={index} className="text-xl font-bold text-indigo-400 mt-6 mb-3">{parseBold(line.slice(4))}</h3>;

        // Numbered Lists (1. )
        if (/^\d+\.\s/.test(line)) {
            return <h3 key={index} className="text-lg font-bold text-white mt-6 mb-3 flex gap-2">
                <span className="text-indigo-500">{line.split('.')[0]}.</span>
                {parseBold(line.replace(/^\d+\.\s/, ''))}
            </h3>;
        }

        // Bullets (•, -, ◦)
        if (line.trim().startsWith('•') || line.trim().startsWith('-')) {
            return <li key={index} className="text-zinc-300 ml-4 mb-2 list-none flex items-start">
                <span className="text-zinc-500 mr-2 mt-1.5 w-1.5 h-1.5 bg-zinc-600 rounded-full shrink-0" />
                <span className="leading-relaxed">{parseBold(line.replace(/^[•-]\s*/, ''))}</span>
            </li>;
        }
        if (line.trim().startsWith('◦')) {
            return <li key={index} className="text-zinc-400 ml-8 mb-1.5 list-none text-sm flex items-start">
                <span className="text-zinc-600 mr-2 mt-2 w-1 h-1 border border-zinc-600 rounded-full shrink-0" />
                <span className="leading-relaxed">{parseBold(line.trim().slice(1))}</span>
            </li>;
        }

        // Empty lines
        if (!line.trim()) return <div key={index} className="h-4" />;

        // Default Paragraph
        return <p key={index} className="text-zinc-300 leading-relaxed mb-2">{parseBold(line)}</p>;
    };

    const parseBold = (text: string) => {
        const parts = text.split(/(\*\*.*?\*\*)/);
        return parts.map((part, i) => {
            if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={i} className="text-white font-bold">{part.slice(2, -2)}</strong>;
            }
            return part;
        });
    };

    if (loading) {
        return (
            <div className="flex h-[50vh] items-center justify-center">
                <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
            </div>
        );
    }

    return (
        <div className="animate-in slide-in-from-right-8 duration-500">
            <button
                onClick={onBack}
                className="mb-6 flex items-center gap-2 text-zinc-400 hover:text-white transition-colors group"
            >
                <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                목록으로 돌아가기
            </button>

            <div className="glass-card bg-[#0a0a0a] border-zinc-800 p-8 md:p-12 max-w-4xl mx-auto shadow-2xl">
                <div className="prose prose-invert max-w-none">
                    {content.split('\n').map((line, i) => parseLine(line, i))}
                </div>
            </div>

            <div className="h-20" /> {/* Bottom spacer */}
        </div>
    );
}
