from CARD.dynamic.dytool import Dytool
from CARD.dynamic.dytool_registry import ToolRegistry


# search
@ToolRegistry.register("Search")
class Dysearch(Dytool):
    def __init__(self):
        super().__init__()
        self.search_mode_info_dict = {
            "Google": "Google is the world's most popular search engine, featuring advanced algorithms, smart suggestions, and vast multilingual content.",
            "Baidu": "Baidu is the top Chinese search engine, known for its strong support for Chinese language, maps, forums, and local services.",
            "DuckDuckGo": "DuckDuckGo is a privacy-first search engine, offering anonymous searching without tracking, ads targeting, or filter bubbles.",
            "Wiki": "Wiki refers to Wikipedia, a user-edited online encyclopedia known for its neutral point of view, openness, and extensive topic coverage.",
        }

        # Search source description mapping
        self.search_source_info_dict = {
            # Developer Communities
            "stackoverflow": "Stack Overflow is the largest Q&A site for developers, offering community-vetted solutions, debugging help, and best practices for programming issues.",
            "github": "GitHub is the leading code hosting platform where you can find open-source projects, source code, technical documentation, and issue discussions.",
            "medium": "Medium is a high-quality blogging platform where developers share deep technical articles, tutorials, and personal insights on tools and trends.",
            "dev": "Dev.to is a modern developer community that provides hands-on tutorials, tech news, and real-world development experiences.",
            "hashnode": "Hashnode is a developer blogging platform where engineers publish technical blogs and share knowledge within the tech community.",
            # Academic Resources
            "arxiv": "arXiv is a preprint server offering cutting-edge research papers, especially in computer science, AI, and physics before peer review.",
            "sciencedirect": "ScienceDirect hosts peer-reviewed scientific papers across disciplines, including computer science, engineering, and life sciences.",
            "springer": "Springer provides comprehensive academic content, including journals, conference papers, and books in science and engineering.",
            "ieee": "IEEE Xplore provides access to high-impact research papers, conference proceedings, and standards in electrical engineering and computing.",
            "acm": "The ACM Digital Library is a premier source for computing research papers, conference proceedings, and technical magazines.",
            "nature": "Nature publishes groundbreaking scientific discoveries across disciplines, offering access to full research articles and expert reviews.",
            "science": "Science is a leading scientific journal that features major research findings, overviews, and scientific commentary across disciplines.",
            # Developer Docs
            "python": "The official Python documentation provides comprehensive references, standard library explanations, tutorials, and usage guidelines.",
            "mozilla": "MDN Web Docs offers in-depth, beginner-to-advanced documentation and examples for HTML, CSS, JavaScript, and Web APIs.",
            "microsoft": "Microsoft Learn is a centralized platform with tutorials, documentation, and interactive learning paths for Microsoft technologies.",
            "aws": "AWS Documentation offers detailed service guides, SDK references, and architecture best practices for Amazon Web Services.",
            "google": "Google Developers provides official docs, code samples, and API references for Google products like Android, Firebase, and Cloud.",
            # Knowledge Bases
            "wikipedia": "Wikipedia is a multilingual, crowd-sourced encyclopedia offering concise, neutral explanations of topics from all domains.",
            "geeksforgeeks": "GeeksforGeeks offers structured tutorials, practice problems, and interview preparation content focused on computer science.",
            "w3schools": "W3Schools provides beginner-friendly tutorials and interactive examples for learning web development technologies.",
            "tutorialspoint": "TutorialsPoint covers a wide range of technologies with step-by-step tutorials, reference guides, and theoretical explanations.",
            # AI/ML Resources
            "huggingface": "Hugging Face hosts pre-trained AI models, datasets, inference tools, and documentation for NLP and machine learning tasks.",
            "pytorch": "PyTorch documentation provides usage guides, API references, and tutorials for building deep learning models with this framework.",
            "tensorflow": "TensorFlow offers comprehensive guides, tutorials, and API documentation for building and deploying ML models.",
            "paperswithcode": "Papers with Code links research papers with implementation code, enabling reproducibility and benchmarking in AI.",
            "distill": "Distill publishes visually rich, intuitive explanations of complex machine learning concepts with an emphasis on clarity and pedagogy.",
            # Q&A Platforms
            "quora": "Quora is a global Q&A platform where experts and enthusiasts share insights and experiences on a wide range of topics.",
            "zhihu": "Zhihu is a Chinese Q&A community offering in-depth discussions, expert answers, and technical problem-solving in Chinese.",
            "reddit": "Reddit hosts diverse tech-focused communities (subreddits) with discussions, tutorials, and real-world developer experiences.",
        }

    def get_info_by_mode(self, mode: str) -> str:
        return self.search_mode_info_dict.get(
            mode, f"Unknown search mode: {mode}. Please check your input."
        )

    def get_info_by_source(self, source: str) -> str:
        return self.search_source_info_dict.get(
            source, f"Unknown search source: {source}. Please check your input."
        )

    def get_dynamic_info(self) -> str:
        return "Dysearch is a tool class for obtaining dynamic information related to search engines/sources. Use `get_info_by_source(source)` method to get specific descriptions."
