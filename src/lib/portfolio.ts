export type PortfolioContent = {
  readonly identity: {
    readonly name: string;
    readonly positioning: readonly string[];
    readonly headline: string;
    readonly introduction: string;
    readonly location: string;
  };
  readonly experience: readonly {
    readonly engagement: string;
    readonly organization: string;
    readonly location: string;
    readonly role: string;
    readonly period: string;
    readonly summary: string;
  }[];
  readonly education: readonly {
    readonly degree: string;
    readonly institution: string;
    readonly period: string;
  }[];
  readonly researchInterests: readonly string[];
  readonly currentResearch: {
    readonly stage: string;
    readonly format: string;
    readonly summary: string;
    readonly questions: readonly string[];
  };
  readonly selectedProject: {
    readonly title: string;
    readonly type: string;
    readonly institution: string;
    readonly advisor: string;
    readonly role: string;
    readonly summary: string;
    readonly technologies: readonly string[];
  };
  readonly technicalExpertise: readonly {
    readonly category: string;
    readonly skills: readonly string[];
  }[];
  readonly links: {
    readonly email: string;
    readonly linkedIn: string;
    readonly projectRepository: string;
  };
};

/**
 * Visitor-facing portfolio content sourced from the attached academic CV.
 * Keep claims here factual and traceable; future content systems can replace
 * this object without changing the presentation components that consume it.
 */
export const portfolioContent = {
  identity: {
    name: "Manikanta Reddy Anugu",
    positioning: [
      "AI/ML Engineer",
      "Aspiring researcher",
      "Prospective PhD student",
    ],
    headline:
      "An AI/ML engineer connecting professional practice with academic inquiry.",
    introduction:
      "I work across applied machine learning and generative AI while developing a research direction focused on how these systems are adopted, evaluated, and sustained in real-world settings.",
    location: "San Antonio, TX",
  },
  experience: [
    {
      engagement: "Client engagement",
      organization: "USAA",
      location: "San Antonio, TX",
      role: "Data Scientist / AI-ML Engineer",
      period: "Jun 2024 - Present",
      summary:
        "Work spanning production LLM pipelines, retrieval-augmented generation, agent workflows, forecasting, MLOps, model monitoring, and data engineering.",
    },
    {
      engagement: "Client engagement",
      organization: "Capita Pvt Ltd",
      location: "Pune, India",
      role: "Data Scientist",
      period: "Jun 2020 - Jul 2022",
      summary:
        "Work across NLP and document processing, forecasting and classification, ML APIs, data pipelines, model interpretability, monitoring, and business intelligence.",
    },
  ],
  education: [
    {
      degree: "Master of Science, Information Technology",
      institution: "Kennesaw State University",
      period: "Aug 2022 - May 2024",
    },
    {
      degree:
        "Bachelor of Technology, Electronics and Communication Engineering",
      institution: "Geethanjali College of Engineering and Technology",
      period: "Jun 2016 - Nov 2020",
    },
  ],
  researchInterests: [
    "Generative AI adoption in industry",
    "LLM comparison and evaluation",
    "Retrieval-augmented generation",
    "The gap between laboratory capability and production reliability",
  ],
  currentResearch: {
    stage: "Early-stage",
    format: "Survey paper through literature and case-study review",
    summary:
      "A review of how companies are using generative AI in practice, which models are seeing real-world adoption, the benefits and constraints organizations encounter, and where the literature leaves open questions.",
    questions: [
      "How are companies using generative AI in practice, and in which functions?",
      "Which LLMs see the most real-world adoption, and why?",
      "What benefits are businesses reporting across cost, speed, and customer experience?",
      "What remains difficult across cost, hallucination, governance, and legacy-system integration?",
      "Where does the literature fall short, and what should researchers examine next?",
    ],
  },
  selectedProject: {
    title:
      "Discover, Learn, and Protect: A Mobile App for Informal STEM Learning about Local Biodiversity and Environmental Issues",
    type: "IT Capstone Project",
    institution: "Kennesaw State University",
    advisor: "Dr. Ying Xie",
    role: "Team Leader and Technical Specialist",
    summary:
      "An interactive educational web application supporting biodiversity and environmental learning through quizzes, video lectures, reading assignments, progress tracking, and AI-assisted learning capabilities.",
    technologies: [
      "HTML",
      "CSS",
      "JavaScript",
      "MongoDB",
      "YouTube API",
      "OpenAI ChatGPT API",
    ],
  },
  technicalExpertise: [
    {
      category: "Programming & Frameworks",
      skills: [
        "Python",
        "PyTorch",
        "TensorFlow",
        "Scikit-learn",
        "XGBoost",
        "LangChain",
        "Hugging Face",
        "JavaScript",
      ],
    },
    {
      category: "AI & Machine Learning",
      skills: [
        "Regression",
        "Classification",
        "Time Series Forecasting",
        "Deep Learning",
        "NLP",
        "RAG",
        "Reinforcement Learning",
        "Generative AI (LLMs)",
      ],
    },
    {
      category: "Data Engineering & Scalability",
      skills: [
        "Apache Spark",
        "Databricks",
        "Snowflake",
        "Kafka",
        "MapReduce",
        "ETL Pipelines",
        "Airflow",
      ],
    },
    {
      category: "MLOps & Deployment",
      skills: [
        "MLflow",
        "Docker",
        "Kubernetes",
        "CI/CD pipelines",
        "SageMaker",
        "Kubeflow",
      ],
    },
    {
      category: "Cloud Platforms",
      skills: ["AWS", "Azure ML", "GCP Vertex AI"],
    },
    {
      category: "Visualization & BI",
      skills: ["Tableau", "Power BI", "Plotly", "Seaborn"],
    },
    {
      category: "Mathematics & Statistics",
      skills: [
        "Linear Algebra",
        "Probability",
        "Statistical Inference",
        "Optimization",
      ],
    },
  ],
  links: {
    email: "mailto:reddymanikanta437@gmail.com",
    linkedIn: "https://www.linkedin.com/in/maniiqe",
    projectRepository: "https://github.com/Manireddy5332/IT-Capstone",
  },
} as const satisfies PortfolioContent;
