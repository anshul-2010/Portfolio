// Project modal data
const projectData = {
  project1: {
    title: "Quantifying and Mitigating Severity Bias in Medical Large Language Models",
    overview: `This research addresses critical bias issues in clinical large language models (LLMs) that can lead to unfair or inaccurate medical assessments. The project investigates how severity bias manifests in medical LLMs when processing oncology narratives from the MIMIC-III dataset, focusing on both fairness and factual fidelity.

    The research develops an Oncology Severity Glossary and implements a graph-based extraction pipeline to map contextual severity patterns in clinical narratives. This enables systematic identification of how severity information influences model predictions and where biases may emerge. The approach combines natural language processing techniques with graph neural networks to capture complex relationships between clinical concepts and severity indicators.

    To mitigate identified biases, the project proposes severity-conditioned attention mechanisms that explicitly account for severity information during model training and inference. Additionally, contrastive representation learning is employed to learn severity-aware embeddings that better distinguish between different severity levels while maintaining clinical accuracy. The framework incorporates severity calibration heads and reward-guided fine-tuning strategies to optimize the trade-off between bias mitigation and model interpretability.`,
    technologies: ["Python", "PyTorch", "Transformers", "HuggingFace", "Graph Neural Networks", "NLP", "MIMIC-III", "Contrastive Learning"],
    results: "Investigating bias in clinical LLMs using oncology narratives from MIMIC-III to assess fairness and factual fidelity. Developed an Oncology Severity Glossary and graph-based extraction pipeline to map contextual severity patterns. Proposed severity-conditioned attention and contrastive representation learning for severity-aware modeling. Incorporating severity calibration heads and reward-guided fine-tuning for bias mitigation and interpretability.",
    repository: "#",
    guide: "Dr. Balaraman Ravindran",
    publication: "Masters Thesis, Centre for Responsible AI (CeRAI), IIT Madras (Jul 2025 - Ongoing)"
  },
  project2: {
    title: "INSIGHT: Multimodal Artifact-Guided Detection of AI-Generated Images",
    overview: `This project addresses the growing challenge of detecting AI-generated images in an era where generative models can produce highly realistic synthetic content. The work was developed as part of the Inter-IIT Tech Meet 13.0 Adobe Research Team AI Challenge, focusing on creating robust detection systems that can identify artifacts and inconsistencies in AI-generated imagery.

    The framework combines visual and linguistic reasoning to achieve comprehensive detection. Visual analysis leverages deep learning models to identify subtle artifacts and inconsistencies that are characteristic of AI-generated images. The linguistic component uses large language models (MOLMO) to provide explainable detection through natural language descriptions of identified artifacts, enhancing both accuracy and interpretability.

    To improve robustness, the system integrates GradCAM-based artifact localization that highlights specific regions where AI-generated artifacts are most likely to occur. This spatial attention mechanism helps focus the model on relevant image regions. The framework also employs adversarial defense ensembles and knowledge distillation techniques to enhance model resilience against adversarial attacks, reducing vulnerability by 21% while maintaining high detection accuracy.`,
    technologies: ["Python", "PyTorch", "GradCAM", "LLMs", "Computer Vision", "Adversarial Defense", "Knowledge Distillation"],
    results: "Achieved 90% accuracy on the CIFAKE perturbation dataset. Integrated GradCAM-based artifact localization with precision of 83%. Enhanced robustness through adversarial defense ensemble, reducing model vulnerability by 21%. Won Bronze Medal in Inter-IIT Tech Meet 13.0.",
    repository: "#",
    publication: "Inter-IIT Tech Meet 13.0, Adobe Research Team AI Challenge (Oct-Dec 2024)"
  },
  project3: {
    title: "Unsupervised Cross-Modality Adaptation for Brain Tumor MRI Segmentation",
    overview: `Medical image segmentation faces significant challenges when models trained on one imaging modality (e.g., contrast-enhanced T1 MRI) need to be applied to different modalities (e.g., high-resolution T2 MRI). This project addresses cross-modality domain shifts that arise from differences in imaging protocols, scanner characteristics, and image appearance.

    The research engineers intensity-mapping techniques and correlation-aware augmentations to counter bias and modality-induced shifts. Intensity normalization methods are developed to align image characteristics across modalities, while correlation-aware augmentations help the model learn modality-invariant features. The approach integrates GIN-IPA (Gaussian Intensity Normalization with Intensity Profile Analysis) for robust intensity normalization.

    A key innovation is the design of causality-guided mechanisms to disentangle spurious correlations that can mislead domain adaptation. By identifying and removing spurious correlations, the model learns more robust cross-modal representations. The framework integrates joint image-feature adaptation with nnUNet architecture, combining image-level transformations with feature-level alignment strategies. This dual-level approach ensures comprehensive adaptation across different imaging modalities.`,
    technologies: ["Python", "PyTorch", "nnUNet", "Medical Image Processing", "Domain Adaptation", "DICOM", "ITK"],
    results: "Achieved Dice scores of 0.63 for Vestibular Schwannoma (VS) and 0.60 for Cochlea segmentation when adapting from contrast-enhanced T1 to high-resolution T2 MRI. The causality-guided mechanisms successfully disentangled spurious correlations, improving cross-modal transfer performance.",
    repository: "#",
    guide: "Dr. Arun K. Thittai",
    publication: "Young Research Fellowship, IIT Madras (Aug 2023 - Aug 2024)"
  },
  project4: {
    title: "Multimodal Simulation of User Behavior and KPI-Driven Content Generation",
    overview: `This project addresses the challenge of predicting user engagement and generating personalized content in dynamic digital environments. The work was developed as part of the Inter-IIT Tech Meet 12.0 Adobe Research MDSR Team AI Challenge, focusing on robust prediction and content generation under cross-brand and temporal domain shifts.

    The framework develops a multi-stage XGBoost pipeline that handles complex feature interactions and temporal dependencies. The model is designed to be robust against domain shifts that occur when applying predictions across different brands or time periods. Feature engineering techniques are employed to capture both static user characteristics and dynamic behavioral patterns.

    For content generation, the system integrates Mistral-7B large language model with LanguageBind embeddings in a KPI-aware Retrieval-Augmented Generation (RAG) framework. This allows the model to generate content that is not only contextually relevant but also optimized for specific key performance indicators. A vector-indexed KPI database is built to enable efficient semantic prompt retrieval using cosine similarity, allowing the system to quickly identify and retrieve the most relevant content templates and examples for generation.`,
    technologies: ["Python", "XGBoost", "Mistral-7B", "LanguageBind", "RAG", "Vector Databases", "NLP"],
    results: "Developed a multi-stage XGBoost pipeline for robust prediction under cross-brand and temporal domain shifts. Integrated Mistral-7B with LanguageBind embeddings in KPI-aware RAG framework for content generation. Built a vector-indexed KPI database enabling efficient semantic prompt retrieval via cosine similarity. Won Bronze Medal in Inter-IIT Tech Meet 12.0.",
    repository: "#",
    publication: "Inter-IIT Tech Meet 12.0, Adobe Research MDSR Team AI Challenge (Oct-Dec 2023)"
  },
  project5: {
    title: "Generative AI-Driven Super-Resolution for Lunar Terrain Mapping",
    overview: `This project addresses the challenge of generating high-resolution lunar terrain maps from lower-resolution satellite imagery. The work was developed as part of the Inter-IIT Tech Meet 11.0 ISRO Chandrayaan-2 Orbiter Imaging AI Challenge, focusing on creating detailed lunar surface maps for scientific and exploration purposes.

    The framework combines two complementary approaches: SRUN (Spatial Attention U-Net) and SORTN (Super-Resolution Optical Terrain Network). SRUN employs spatial attention mechanisms to focus on important terrain features during upscaling, while SORTN handles optical characteristics specific to lunar imaging. The system is designed to generate 30 cm-resolution images from 10 m TMC (Terrain Mapping Camera) data, representing a significant improvement in spatial resolution.

    Adaptive histogram scaling techniques are implemented to ensure high-fidelity image reconstruction while preserving important terrain characteristics. The framework processes multiple spectral bands and integrates them to create comprehensive terrain maps. The system is trained on paired low-resolution and high-resolution lunar imagery, learning to reconstruct fine details such as craters, ridges, and surface textures that are critical for lunar exploration and scientific analysis.`,
    technologies: ["Python", "PyTorch", "U-Net", "Spatial Attention", "Image Super-Resolution", "Remote Sensing", "Computer Vision"],
    results: "Developed a super-resolution framework (SRUN + SORTN) to generate 30 cm-resolution from 10 m TMC data. Implemented spatial attention U-Nets and adaptive histogram scaling to ensure high-fidelity image reconstruction. Achieved PSNR 28.26, SSIM 0.79 at 4× upscaling, enabling the generation of preliminary AI-based lunar atlas.",
    repository: "#",
    publication: "Inter-IIT Tech Meet 11.0, ISRO Chandrayaan-2 Orbiter Imaging AI Challenge (Sept 2022 - Feb 2023)"
  },
  project6: {
    title: "Domain Invariant Multi-Organ Segmentation via Contrastive Adaptation",
    overview: `Medical image segmentation faces significant challenges when models trained on data from one scanner or institution are applied to images from different sources. This project investigates test-time domain adaptation frameworks for multi-organ segmentation under cross-scanner shifts, where differences in imaging protocols, scanner manufacturers, and acquisition parameters can significantly degrade model performance.

    The research integrates contrastive alignment techniques with transformer encoders to learn domain-invariant anatomical features. Contrastive learning is used to pull together representations of the same anatomical structures across different domains while pushing apart representations of different structures, regardless of domain. This approach helps the model focus on anatomical consistency rather than scanner-specific characteristics.

    Transformer encoders are employed to capture long-range dependencies and contextual information that are crucial for accurate organ segmentation. The framework operates at test time, meaning it can adapt to new domains without requiring retraining on domain-specific data. This makes it particularly valuable for clinical deployment where scanner diversity is common. The system processes CT scans and segments multiple organs simultaneously, learning to handle variations in image appearance, contrast, and noise levels across different scanners.`,
    technologies: ["Python", "PyTorch", "Transformers", "Contrastive Learning", "Medical Image Processing", "Domain Adaptation", "Multi-Organ Segmentation"],
    results: "Investigated test-time domain adaptation frameworks for multi-organ segmentation under cross-scanner shifts. Integrated contrastive alignment with transformer encoders to learn domain-invariant anatomical features. Achieved steady Dice scores (0.54-0.80) across multi-organ CT segmentation benchmarks despite domain disparity.",
    repository: "#",
    guide: "Dr. Vaanathi Sundaresan",
    publication: "Biomedical Image Analysis (BioMedIA) Laboratory, IISc, Bengaluru (May 2023 - Dec 2023)"
  },
  project7: {
    title: "Causality Driven Uplift Modeling for Customer Engagement Optimization",
    overview: `This project applies causal inference frameworks to estimate the impact of business interventions on customer behavior, specifically focusing on the LifeSync program's effect on advisor bookings. The work was conducted at Wells Fargo's Consumer Model Development Center, addressing the challenge of measuring true causal effects rather than mere correlations in business analytics.

    The framework employs causal inference techniques to estimate treatment effects, accounting for confounding variables that could bias traditional observational analyses. Conditional Average Treatment Effect (CATE) models are used to identify heterogeneous treatment effects, revealing that different customer segments respond differently to the intervention. This allows for more targeted and efficient policy implementation.

    Policy optimization strategies are formulated based on the estimated treatment effects, identifying high-response customer subgroups and tailoring interventions accordingly. The framework includes ROI simulation capabilities that demonstrate potential efficiency gains through targeted causal policy evaluation. The system estimates not just whether an intervention works, but for whom it works best, enabling data-driven decision making in customer engagement strategies.`,
    technologies: ["Python", "Causal Inference", "CATE Models", "Uplift Modeling", "Business Analytics", "Statistical Modeling", "Policy Optimization"],
    results: "Applied causal inference frameworks to estimate LifeSync's impact, revealing a 3-5% uplift in advisor bookings. Estimated heterogeneous treatment effects via CATE models to identify high-response customer subgroups. Formulated policy optimization strategies achieving up to 15% higher engagement in top-tier segments. Demonstrated potential 2-3× efficiency gains through targeted causal policy evaluation and ROI simulation.",
    repository: "#",
    publication: "Consumer Model Development Center, Business Analytics, Wells Fargo (May 2025 - Jul 2025), Bengaluru, Karnataka, India"
  },
  project8: {
    title: "Multimodal Learning for AI-Driven Diabetic Retinopathy Diagnosis",
    overview: `This project addresses the challenge of diagnosing diabetic retinopathy (DR) by combining multiple data modalities to improve diagnostic accuracy. The work was conducted during an ML Developer Internship at SiddhaAI, focusing on creating a comprehensive diagnostic framework that leverages both retinal imaging and physiological signals.

    The framework combines retinal fundus images with vital sign data from ICU monitors to provide a more complete picture of patient health. Retinal images are processed using deep learning models to identify DR-related lesions and abnormalities, while physiological signals provide additional context about the patient's overall health status. This multimodal approach helps overcome limitations of single-modality diagnosis.

    Semi-supervised learning with SCAN (Semantic Clustering by Adopting Nearest neighbors) is leveraged for efficient label propagation, enhancing generalization on limited clinical data. This is particularly important in medical AI where labeled data is often scarce and expensive to obtain. The system uses YOLOv8 for object detection in vital sign monitor displays, achieving high accuracy in extracting relevant physiological parameters. OCR-based text recognition is employed to read monitor displays and extract numerical values, which are then integrated with image-based features for comprehensive diagnosis.`,
    technologies: ["Python", "PyTorch", "YOLOv8", "OCR", "Semi-supervised Learning", "SCAN", "Multimodal Learning", "Medical AI"],
    results: "Built a multimodal diagnostic framework combining retinal imaging and physiological signals for DR detection. Leveraged semi-supervised SCAN for efficient label propagation, enhancing generalization on limited clinical data. Achieved mAP50 > 0.98 in vital sign extraction using YOLOv8 and OCR-based monitor text recognition.",
    repository: "#",
    publication: "ML Developer Intern, SiddhaAI, McKinney, Texas, USA (Jan 2022 - Mar 2023)"
  },
  project9: {
    title: "Automated Microscopic Phenotyping for Arabidopsis Seeds",
    overview: `This project addresses the need for high-throughput phenotyping in plant biology research, specifically focusing on automated classification and analysis of Arabidopsis seed lines. The work was conducted in the Developmental Genetics Laboratory at IIT Madras, supporting research in plant development and genetics.

    The framework develops an automated imaging pipeline that processes microscopic images of Arabidopsis seeds and classifies them based on phenotypic characteristics. The system is designed to handle high-throughput analysis, processing large numbers of seed samples efficiently. This automation significantly reduces the time and labor required for manual phenotyping while improving consistency and reproducibility.

    OpenCV-based segmentation techniques are employed to isolate individual seeds from images and identify key morphological features. Watershed clustering algorithms are used to separate touching or overlapping seeds, ensuring accurate individual seed analysis. The system extracts various phenotypic features including size, shape, color, and texture characteristics that are important for genetic studies.

    A scalable Python automation tool is engineered to handle the entire pipeline from image acquisition to phenotype labeling. The system includes quality control mechanisms to identify and flag problematic images or segmentation results, ensuring reliable data for downstream genetic analysis.`,
    technologies: ["Python", "OpenCV", "Image Segmentation", "Watershed Clustering", "Computer Vision", "Bioinformatics", "Automation"],
    results: "Developed an automated imaging pipeline for phenotypic classification of Arabidopsis seed lines in MeioSeed. Implemented OpenCV-based segmentation and watershed clustering, achieving 94.5% accuracy (AUC: 0.75). Engineered a scalable Python automation tool for high-throughput seed counting and phenotype labeling.",
    repository: "#",
    guide: "Prof. R Baskar",
    publication: "Developmental Genetics Laboratory, Biotechnology Department, IIT Madras (Sept 2022 - Apr 2023)"
  }
};

// Open project modal
function openProjectModal(projectId) {
  const project = projectData[projectId];
  if (!project) return;

  const modal = document.getElementById('project-modal');
  const content = document.getElementById('modal-content');
  
  content.innerHTML = `
    <h2 class="text-2xl sm:text-3xl font-semibold text-slate-900 mb-4 leading-tight">${project.title}</h2>
    ${project.guide ? `<p class="text-sm text-slate-600 mb-6 font-medium">Guide: ${project.guide}</p>` : ''}
    
    <div class="mb-8">
      <h3 class="text-lg font-semibold text-slate-900 mb-4">Project Overview</h3>
      <p class="text-sm leading-relaxed text-slate-700 whitespace-pre-line">${project.overview}</p>
    </div>
    
    <div class="mb-8">
      <h3 class="text-lg font-semibold text-slate-900 mb-4">Technical Implementation</h3>
      <ul class="list-disc list-inside text-sm text-slate-700 space-y-2 ml-2">
        ${project.technologies.map(tech => `<li>${tech}</li>`).join('')}
      </ul>
    </div>
    
    <div class="mb-8">
      <h3 class="text-lg font-semibold text-slate-900 mb-4">Key Results</h3>
      <p class="text-sm leading-relaxed text-slate-700">${project.results}</p>
    </div>
    
    <div class="flex flex-wrap gap-3 pt-6 border-t border-slate-200">
      <a href="${project.repository}" target="_blank" class="inline-flex items-center px-5 py-2.5 bg-slate-900 text-white rounded-md hover:bg-slate-800 transition-all text-sm font-medium">
        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        View Code Repository
      </a>
      <a href="#" class="inline-flex items-center px-5 py-2.5 bg-white text-slate-700 rounded-md hover:bg-slate-50 transition-all text-sm font-medium border border-slate-200">
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
        </svg>
        Related Publication
      </a>
    </div>
  `;
  
  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

// Close project modal
function closeProjectModal() {
  const modal = document.getElementById('project-modal');
  modal.classList.add('hidden');
  document.body.style.overflow = 'auto';
}

// Close modal when clicking outside
document.addEventListener('DOMContentLoaded', function() {
  const modal = document.getElementById('project-modal');
  if (modal) {
    modal.addEventListener('click', function(e) {
      if (e.target === this) {
        closeProjectModal();
      }
    });
  }

  // Close modal with Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      closeProjectModal();
    }
  });
});

