// `seedKey` values are immutable bootstrap identities. Public slugs may change
// later without changing which database record an explicit seed run reconciles.
export const portfolioSeed = {
  profile: {
    key: "primary",
    slug: "manikanta-reddy-anugu",
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
    about: [
      "My professional path spans data science and AI/ML engineering. I worked as a Data Scientist on a client engagement with Capita Pvt Ltd and currently work as a Data Scientist / AI-ML Engineer on a client engagement with USAA.",
      "Across these roles, my work has included machine learning, generative AI, retrieval-augmented generation, forecasting, NLP, MLOps, model monitoring, and data engineering.",
      "Alongside professional practice, I am developing an early-stage survey paper through literature and case-study review focused on how generative AI is adopted, evaluated, and sustained in real-world organizations.",
    ],
    location: "San Antonio, TX",
  },
  socialLinks: [
    {
      key: "email",
      kind: "EMAIL",
      label: "Email",
      url: "mailto:reddymanikanta437@gmail.com",
      sortOrder: 0,
    },
    {
      key: "linkedin",
      kind: "LINKEDIN",
      label: "LinkedIn",
      url: "https://www.linkedin.com/in/maniiqe",
      sortOrder: 1,
    },
  ],
  researchInterests: [
    {
      seedKey: "generative-ai-adoption-in-industry",
      slug: "generative-ai-adoption-in-industry",
      name: "Generative AI adoption in industry",
      sortOrder: 0,
    },
    {
      seedKey: "llm-comparison-and-evaluation",
      slug: "llm-comparison-and-evaluation",
      name: "LLM comparison and evaluation",
      sortOrder: 1,
    },
    {
      seedKey: "retrieval-augmented-generation",
      slug: "retrieval-augmented-generation",
      name: "Retrieval-augmented generation",
      sortOrder: 2,
    },
    {
      seedKey: "lab-to-production-gap",
      slug: "lab-to-production-gap",
      name: "The gap between laboratory capability and production reliability",
      sortOrder: 3,
    },
  ],
  researchProject: {
    seedKey: "generative-ai-adoption-in-industry",
    slug: "generative-ai-adoption-in-industry",
    title: "Generative AI adoption in industry",
    summary:
      "A review of how companies are using generative AI in practice, which models are seeing real-world adoption, the benefits and constraints organizations encounter, and where the literature leaves open questions.",
    format: "Survey paper through literature and case-study review",
    methodology: "Literature and case-study review",
    methodologySummary:
      "The review examines how companies use generative AI, which models are adopted, and what benefits and constraints are reported.",
    scopeBoundary:
      "This review examines existing literature and case studies. It is not a project to build or train a new model.",
    evidenceStatus:
      "The review is in its early stage. No findings or conclusions are presented.",
    questions: [
      {
        key: "industry-use-and-functions",
        question:
          "How are companies using generative AI in practice, and in which functions?",
        sortOrder: 0,
      },
      {
        key: "llm-adoption-and-rationale",
        question: "Which LLMs see the most real-world adoption, and why?",
        sortOrder: 1,
      },
      {
        key: "reported-business-benefits",
        question:
          "What benefits are businesses reporting across cost, speed, and customer experience?",
        sortOrder: 2,
      },
      {
        key: "operational-constraints",
        question:
          "What remains difficult across cost, hallucination, governance, and legacy-system integration?",
        sortOrder: 3,
      },
      {
        key: "literature-gaps-and-next-questions",
        question:
          "Where does the literature fall short, and what should researchers examine next?",
        sortOrder: 4,
      },
    ],
  },
  project: {
    seedKey: "discover-learn-and-protect",
    slug: "discover-learn-and-protect",
    title:
      "Discover, Learn, and Protect: A Mobile App for Informal STEM Learning about Local Biodiversity and Environmental Issues",
    shortTitle: "Discover, Learn, and Protect",
    type: "IT Capstone Project",
    category: "Academic capstone",
    summary:
      "An interactive educational web application supporting biodiversity and environmental learning through quizzes, video lectures, reading assignments, progress tracking, and AI-assisted learning capabilities.",
    institution: "Kennesaw State University",
    advisor: "Dr. Ying Xie",
    role: "Team Leader and Technical Specialist",
    implementation: "Interactive educational web application",
    repositoryUrl: "https://github.com/Manireddy5332/IT-Capstone",
    contributions: [
      {
        key: "leadership-and-coordination",
        label: "Leadership and coordination",
        description:
          "Coordinated project planning, task allocation, weekly meetings, communication with the project owner, and support across the team lifecycle.",
        sortOrder: 0,
      },
      {
        key: "application-development",
        label: "Application development",
        description:
          "Contributed to front-end and back-end development, database design, feature implementation, and integration of application components.",
        sortOrder: 1,
      },
      {
        key: "learning-experience",
        label: "Learning experience",
        description:
          "Helped shape quizzes, video lectures, reading assignments, progress tracking, and AI-assisted learning for biodiversity education.",
        sortOrder: 2,
      },
      {
        key: "integrations-and-quality",
        label: "Integrations and quality",
        description:
          "Integrated YouTube and OpenAI ChatGPT APIs and participated in testing, debugging, feature enhancement, and the final presentation.",
        sortOrder: 3,
      },
    ],
    features: [
      { key: "quizzes", name: "Quizzes", sortOrder: 0 },
      { key: "video-lectures", name: "Video lectures", sortOrder: 1 },
      {
        key: "reading-assignments",
        name: "Reading assignments",
        sortOrder: 2,
      },
      {
        key: "progress-tracking",
        name: "Progress tracking",
        sortOrder: 3,
      },
      {
        key: "ai-assisted-learning-capabilities",
        name: "AI-assisted learning capabilities",
        sortOrder: 4,
      },
    ],
    technologies: [
      { slug: "html", name: "HTML", sortOrder: 0 },
      { slug: "css", name: "CSS", sortOrder: 1 },
      { slug: "javascript", name: "JavaScript", sortOrder: 2 },
      { slug: "mongodb", name: "MongoDB", sortOrder: 3 },
      { slug: "youtube-api", name: "YouTube API", sortOrder: 4 },
      {
        slug: "openai-chatgpt-api",
        name: "OpenAI ChatGPT API",
        sortOrder: 5,
      },
    ],
  },
  experiences: [
    {
      seedKey: "usaa-client-engagement",
      slug: "usaa-client-engagement",
      organization: "USAA",
      role: "Data Scientist / AI-ML Engineer",
      location: "San Antonio, TX",
      summary:
        "Work spanning production LLM pipelines, retrieval-augmented generation, agent workflows, forecasting, MLOps, model monitoring, and data engineering.",
      practiceAreas: [
        "Generative AI and RAG",
        "Agent workflows",
        "Forecasting",
        "MLOps and monitoring",
        "Data engineering",
      ],
      startYear: 2024,
      startMonth: 6,
      endYear: null,
      endMonth: null,
      isCurrent: true,
      sortOrder: 0,
    },
    {
      seedKey: "capita-pvt-ltd-client-engagement",
      slug: "capita-pvt-ltd-client-engagement",
      organization: "Capita Pvt Ltd",
      role: "Data Scientist",
      location: "Pune, India",
      summary:
        "Work across NLP and document processing, forecasting and classification, ML APIs, data pipelines, model interpretability, monitoring, and business intelligence.",
      practiceAreas: [
        "NLP and document processing",
        "Forecasting and classification",
        "ML APIs",
        "Model interpretability",
        "Business intelligence",
      ],
      startYear: 2020,
      startMonth: 6,
      endYear: 2022,
      endMonth: 7,
      isCurrent: false,
      sortOrder: 1,
    },
  ],
  education: [
    {
      seedKey: "kennesaw-state-ms-information-technology",
      slug: "kennesaw-state-ms-information-technology",
      institution: "Kennesaw State University",
      degree: "Master of Science, Information Technology",
      startYear: 2022,
      startMonth: 8,
      endYear: 2024,
      endMonth: 5,
      sortOrder: 0,
    },
    {
      seedKey: "geethanjali-btech-electronics-communication-engineering",
      slug: "geethanjali-btech-electronics-communication-engineering",
      institution: "Geethanjali College of Engineering and Technology",
      degree:
        "Bachelor of Technology, Electronics and Communication Engineering",
      startYear: 2016,
      startMonth: 6,
      endYear: 2020,
      endMonth: 11,
      sortOrder: 1,
    },
  ],
  skillCategories: [
    {
      seedKey: "programming-frameworks",
      slug: "programming-frameworks",
      name: "Programming & Frameworks",
      sortOrder: 0,
      skills: [
        { seedKey: "python", slug: "python", name: "Python" },
        { seedKey: "pytorch", slug: "pytorch", name: "PyTorch" },
        { seedKey: "tensorflow", slug: "tensorflow", name: "TensorFlow" },
        {
          seedKey: "scikit-learn",
          slug: "scikit-learn",
          name: "Scikit-learn",
        },
        { seedKey: "xgboost", slug: "xgboost", name: "XGBoost" },
        { seedKey: "langchain", slug: "langchain", name: "LangChain" },
        {
          seedKey: "hugging-face",
          slug: "hugging-face",
          name: "Hugging Face",
        },
        {
          seedKey: "javascript",
          slug: "javascript",
          name: "JavaScript",
        },
      ],
    },
    {
      seedKey: "ai-machine-learning",
      slug: "ai-machine-learning",
      name: "AI & Machine Learning",
      sortOrder: 1,
      skills: [
        { seedKey: "regression", slug: "regression", name: "Regression" },
        {
          seedKey: "classification",
          slug: "classification",
          name: "Classification",
        },
        {
          seedKey: "time-series-forecasting",
          slug: "time-series-forecasting",
          name: "Time Series Forecasting",
        },
        {
          seedKey: "deep-learning",
          slug: "deep-learning",
          name: "Deep Learning",
        },
        { seedKey: "nlp", slug: "nlp", name: "NLP" },
        { seedKey: "rag", slug: "rag", name: "RAG" },
        {
          seedKey: "reinforcement-learning",
          slug: "reinforcement-learning",
          name: "Reinforcement Learning",
        },
        {
          seedKey: "generative-ai-llms",
          slug: "generative-ai-llms",
          name: "Generative AI (LLMs)",
        },
      ],
    },
    {
      seedKey: "data-engineering-scalability",
      slug: "data-engineering-scalability",
      name: "Data Engineering & Scalability",
      sortOrder: 2,
      skills: [
        {
          seedKey: "apache-spark",
          slug: "apache-spark",
          name: "Apache Spark",
        },
        {
          seedKey: "databricks",
          slug: "databricks",
          name: "Databricks",
        },
        { seedKey: "snowflake", slug: "snowflake", name: "Snowflake" },
        { seedKey: "kafka", slug: "kafka", name: "Kafka" },
        { seedKey: "mapreduce", slug: "mapreduce", name: "MapReduce" },
        {
          seedKey: "etl-pipelines",
          slug: "etl-pipelines",
          name: "ETL Pipelines",
        },
        { seedKey: "airflow", slug: "airflow", name: "Airflow" },
      ],
    },
    {
      seedKey: "mlops-deployment",
      slug: "mlops-deployment",
      name: "MLOps & Deployment",
      sortOrder: 3,
      skills: [
        { seedKey: "mlflow", slug: "mlflow", name: "MLflow" },
        { seedKey: "docker", slug: "docker", name: "Docker" },
        {
          seedKey: "kubernetes",
          slug: "kubernetes",
          name: "Kubernetes",
        },
        {
          seedKey: "ci-cd-pipelines",
          slug: "ci-cd-pipelines",
          name: "CI/CD pipelines",
        },
        { seedKey: "sagemaker", slug: "sagemaker", name: "SageMaker" },
        { seedKey: "kubeflow", slug: "kubeflow", name: "Kubeflow" },
      ],
    },
    {
      seedKey: "cloud-platforms",
      slug: "cloud-platforms",
      name: "Cloud Platforms",
      sortOrder: 4,
      skills: [
        { seedKey: "aws", slug: "aws", name: "AWS" },
        { seedKey: "azure-ml", slug: "azure-ml", name: "Azure ML" },
        {
          seedKey: "gcp-vertex-ai",
          slug: "gcp-vertex-ai",
          name: "GCP Vertex AI",
        },
      ],
    },
    {
      seedKey: "visualization-bi",
      slug: "visualization-bi",
      name: "Visualization & BI",
      sortOrder: 5,
      skills: [
        { seedKey: "tableau", slug: "tableau", name: "Tableau" },
        { seedKey: "power-bi", slug: "power-bi", name: "Power BI" },
        { seedKey: "plotly", slug: "plotly", name: "Plotly" },
        { seedKey: "seaborn", slug: "seaborn", name: "Seaborn" },
      ],
    },
    {
      seedKey: "mathematics-statistics",
      slug: "mathematics-statistics",
      name: "Mathematics & Statistics",
      sortOrder: 6,
      skills: [
        {
          seedKey: "linear-algebra",
          slug: "linear-algebra",
          name: "Linear Algebra",
        },
        {
          seedKey: "probability",
          slug: "probability",
          name: "Probability",
        },
        {
          seedKey: "statistical-inference",
          slug: "statistical-inference",
          name: "Statistical Inference",
        },
        {
          seedKey: "optimization",
          slug: "optimization",
          name: "Optimization",
        },
      ],
    },
  ],
} as const;
