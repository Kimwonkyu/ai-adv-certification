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

    const parseContent = (text: string) => {
        const lines = text.split('\n');
        const elements: React.ReactNode[] = [];
        let codeBlock: string[] = [];
        let isCodeBlock = false;
        let codeLanguage = '';

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Code Block Start/End
            if (line.trim().startsWith('```')) {
                if (!isCodeBlock) {
                    isCodeBlock = true;
                    codeLanguage = line.trim().slice(3) || 'python';
                    codeBlock = [];
                } else {
                    isCodeBlock = false;
                    elements.push(
                        <div key={`code-${i}`} className="my-6 rounded-xl overflow-hidden border border-zinc-800 bg-[#0d0d0d] shadow-2xl">
                            <div className="flex items-center justify-between px-4 py-2 bg-zinc-900/50 border-b border-zinc-800">
                                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{codeLanguage}</span>
                                <div className="flex gap-1.5">
                                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                                </div>
                            </div>
                            <pre className="p-4 overflow-x-auto text-sm leading-relaxed font-mono text-zinc-300">
                                <code>{codeBlock.join('\n')}</code>
                            </pre>
                        </div>
                    );
                }
                continue;
            }

            if (isCodeBlock) {
                codeBlock.push(line);
                continue;
            }

            // Horizontal Rule
            if (line.trim() === '---') {
                elements.push(<hr key={i} className="my-10 border-zinc-800" />);
                continue;
            }

            // Headers
            if (line.startsWith('# ')) {
                elements.push(<h1 key={i} className="text-3xl font-black text-white mt-12 mb-8 pb-4 border-b-2 border-indigo-500/20">{renderInline(line.slice(2))}</h1>);
                continue;
            }
            if (line.startsWith('## ')) {
                elements.push(<h2 key={i} className="text-2xl font-bold text-white mt-10 mb-6 flex items-center gap-3">
                    <span className="w-1.5 h-6 bg-indigo-500 rounded-full" />
                    {renderInline(line.slice(3))}
                </h2>);
                continue;
            }
            if (line.startsWith('### ')) {
                elements.push(<h3 key={i} className="text-xl font-bold text-indigo-400 mt-8 mb-4">{renderInline(line.slice(4))}</h3>);
                continue;
            }

            // Numbered Lists
            if (/^\d+\.\s/.test(line)) {
                elements.push(
                    <div key={i} className="text-lg font-bold text-white mt-8 mb-4 flex gap-3 items-baseline">
                        <span className="text-indigo-500 tabular-nums">{line.split('.')[0]}.</span>
                        <div>{renderInline(line.replace(/^\d+\.\s/, ''))}</div>
                    </div>
                );
                continue;
            }

            // Bullets
            if (line.trim().startsWith('•') || line.trim().startsWith('- ')) {
                elements.push(
                    <div key={i} className="text-zinc-300 ml-4 mb-3 list-none flex items-start group">
                        <span className="text-indigo-500/50 mr-3 mt-2 w-1.5 h-1.5 bg-indigo-500 rounded-full shrink-0 group-hover:scale-125 transition-transform" />
                        <span className="leading-relaxed">{renderInline(line.replace(/^[•-]\s*/, ''))}</span>
                    </div>
                );
                continue;
            }
            if (line.trim().startsWith('◦')) {
                elements.push(
                    <div key={i} className="text-zinc-400 ml-10 mb-2 list-none text-sm flex items-start">
                        <span className="text-zinc-600 mr-2 mt-2 w-1 h-1 border border-zinc-600 rounded-full shrink-0" />
                        <span className="leading-relaxed">{renderInline(line.trim().slice(1))}</span>
                    </div>
                );
                continue;
            }

            // Empty lines
            if (!line.trim()) {
                elements.push(<div key={i} className="h-4" />);
                continue;
            }

            // Default Paragraph
            elements.push(<p key={i} className="text-zinc-300 leading-relaxed mb-4 text-justify">{renderInline(line)}</p>);
        }

        return elements;
    };

    const renderInline = (text: string) => {
        // First handle bold **text**
        let parts = text.split(/(\*\*.*?\*\*)/);
        let elements: React.ReactNode[] = parts.map((part, i) => {
            if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={`bold-${i}`} className="text-white font-bold bg-zinc-800/50 px-1 rounded">{part.slice(2, -2)}</strong>;
            }
            return part;
        });

        // Then handle inline code `code`
        return elements.map((element, i) => {
            if (typeof element !== 'string') return element;

            const subParts = element.split(/(`.*?`)/);
            return subParts.map((subPart, j) => {
                if (subPart.startsWith('`') && subPart.endsWith('`')) {
                    return <code key={`code-${i}-${j}`} className="text-indigo-300 bg-indigo-500/10 px-1.5 py-0.5 rounded font-mono text-[0.9em] border border-indigo-500/20">{subPart.slice(1, -1)}</code>;
                }
                return subPart;
            });
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
                    {parseContent(content)}
                </div>
            </div>

            <div className="h-20" /> {/* Bottom spacer */}
        </div>
    );
}
