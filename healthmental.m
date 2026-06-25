% ============================================================
%  SNS 사용량에 따른 청소년 우울 위험도 예측
%  머신러닝 모델: SVM (RBF Kernel)
%  데이터: Teen Mental Health Dataset (N=2500)
% ============================================================
clear; clc; close all;

%% ── 0. 공통 색상 팔레트 설정 (발표용) ─────────────────────────
COLOR_LOW  = [0.20, 0.60, 0.86];   % 파랑  – Low risk
COLOR_MED  = [0.98, 0.76, 0.18];   % 노랑  – Medium risk
COLOR_HIGH = [0.90, 0.30, 0.24];   % 빨강  – High risk
PALETTE    = [COLOR_LOW; COLOR_MED; COLOR_HIGH];
BG_COLOR   = [0.97, 0.97, 0.97];

fprintf('==============================================\n');
fprintf('  SNS 사용량에 따른 청소년 우울 위험도 예측\n');
fprintf('         모델: SVM (RBF Kernel)\n');
fprintf('==============================================\n\n');

% ============================================================
%% 1. 데이터 로드
% ============================================================
fprintf('[1/5] 데이터 로드 중...\n');
data = readtable("Teen_Mental_Health_Dataset.csv", ...
                 "TextType", "string");
fprintf('  - 전체 샘플 수: %d\n', height(data));

% ============================================================
%% 2. 데이터 전처리
% ============================================================
fprintf('[2/5] 데이터 전처리 중...\n');

% 사용 변수 선택
feat_cols   = {'daily_social_media_hours', 'stress_level', 'anxiety_level', ...
               'sleep_hours', 'academic_performance'};
feat_labels = {'SNS Usage (h)', 'Stress Level', 'Anxiety Level', ...
               'Sleep Time (h)', 'GPA'};

% 특성 행렬 추출
X = table2array(data(:, feat_cols));

% 타겟 인코딩: low=1, medium=2, high=3
risk_str = lower(string(data.depression_risk));
y = zeros(height(data), 1);
y(risk_str == "low")    = 1;
y(risk_str == "medium") = 2;
y(risk_str == "high")   = 3;
class_names = {'Low', 'Medium', 'High'};

fprintf('  - 클래스 분포: Low=%d / Medium=%d / High=%d\n', ...
        sum(y==1), sum(y==2), sum(y==3));

% Train(2000) / Test(500) 분리
rng(42);
idx      = randperm(height(data));
trainIdx = idx(1:2000);
testIdx  = idx(2001:2500);

X_train = X(trainIdx,:);  y_train = y(trainIdx);
X_test  = X(testIdx, :);  y_test  = y(testIdx);

fprintf('  - Train: %d / Test: %d\n', length(y_train), length(y_test));

% Z-score 정규화 (train 통계 기준)
mu    = mean(X_train);
sigma = std(X_train);
X_tr  = (X_train - mu) ./ sigma;
X_te  = (X_test  - mu) ./ sigma;

fprintf('  - 정규화 완료 (Z-score)\n\n');

% ============================================================
%% 3. 상관관계 분석 – Heatmap
% ============================================================
fprintf('[3/5] 상관관계 분석 중...\n');

corr_data   = [X, y];
corr_labels = [feat_labels, {'Depression Risk'}];
R = corr(corr_data, 'Rows', 'complete');

figure('Name', 'Correlation Heatmap', ...
       'Position', [50, 50, 720, 620], 'Color', 'white');

% Red-Blue diverging colormap 생성
n = 256;
r = [linspace(0.13, 1, n/2), ones(1, n/2)];
g = [linspace(0.27, 1, n/2), linspace(1, 0.18, n/2)];
b = [ones(1, n/2),            linspace(1, 0.18, n/2)];
cmap_rb = [r', g', b'];

imagesc(R);
colormap(cmap_rb);
clim([-1, 1]);
cb = colorbar;
cb.Label.String   = 'Pearson r';
cb.Label.FontSize = 11;

xticks(1:length(corr_labels));  xticklabels(corr_labels);  xtickangle(30);
yticks(1:length(corr_labels));  yticklabels(corr_labels);

% 셀 내 수치 표기
for i = 1:size(R,1)
    for j = 1:size(R,2)
        val = R(i,j);
        if abs(val) > 0.45
            txtColor = 'white';
        else
            txtColor = [0.2, 0.2, 0.2];
        end
        text(j, i, sprintf('%.2f', val), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment',   'middle', ...
            'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', txtColor);
    end
end

title('Feature Correlation Heatmap', 'FontSize', 15, 'FontWeight', 'bold');
set(gca, 'FontSize', 11, 'TickLength', [0,0]);
box off;
fprintf('  - Heatmap 완료\n\n');

% ============================================================
%% 4. 탐색적 시각화 (EDA Scatter Plots)
% ============================================================
fprintf('[4/5] EDA 시각화 중...\n');

% ── Scatter 1: SNS 사용량 vs 스트레스 ─────────────────────────
figure('Name', 'Scatter: SNS Usage vs Stress', ...
       'Position', [50, 50, 800, 580], 'Color', 'white');

h_grp = gobjects(3,1);
for k = 1:3
    mask = (y == k);
    h_grp(k) = scatter(X(mask,1), X(mask,2), 45, PALETTE(k,:), ...
        'filled', 'MarkerFaceAlpha', 0.65, 'DisplayName', class_names{k});
    hold on;
end
for k = 1:3
    mask = (y == k);
    p  = polyfit(X(mask,1), X(mask,2), 1);
    xr = linspace(min(X(mask,1)), max(X(mask,1)), 100);
    plot(xr, polyval(p,xr), '-', 'Color', PALETTE(k,:), ...
        'LineWidth', 2.2, 'HandleVisibility', 'off');
end
hold off;

xlabel('SNS Daily Usage Time (hours)', 'FontSize', 13);
ylabel('Stress Level',                 'FontSize', 13);
title('SNS Usage Time vs Stress Level', 'FontSize', 15, 'FontWeight', 'bold');
lgd1 = legend(h_grp, class_names, 'Location', 'northwest', 'FontSize', 11);
lgd1.Title.String = 'Depression Risk';
set(gca, 'FontSize', 11, 'Color', BG_COLOR, 'Box', 'off');
grid on; grid minor;

% ── Scatter 2: SNS 사용량 vs 수면 시간 ────────────────────────
figure('Name', 'Scatter: SNS Usage vs Sleep', ...
       'Position', [100, 100, 800, 580], 'Color', 'white');

h_grp2 = gobjects(3,1);
for k = 1:3
    mask = (y == k);
    h_grp2(k) = scatter(X(mask,1), X(mask,4), 45, PALETTE(k,:), ...
        'filled', 'MarkerFaceAlpha', 0.65, 'DisplayName', class_names{k});
    hold on;
end
for k = 1:3
    mask = (y == k);
    p  = polyfit(X(mask,1), X(mask,4), 1);
    xr = linspace(min(X(mask,1)), max(X(mask,1)), 100);
    plot(xr, polyval(p,xr), '-', 'Color', PALETTE(k,:), ...
        'LineWidth', 2.2, 'HandleVisibility', 'off');
end
hold off;

xlabel('SNS Daily Usage Time (hours)', 'FontSize', 13);
ylabel('Sleep Time (hours)',           'FontSize', 13);
title('SNS Usage Time vs Sleep Time', 'FontSize', 15, 'FontWeight', 'bold');
lgd2 = legend(h_grp2, class_names, 'Location', 'northeast', 'FontSize', 11);
lgd2.Title.String = 'Depression Risk';
set(gca, 'FontSize', 11, 'Color', BG_COLOR, 'Box', 'off');
grid on; grid minor;

fprintf('  - Scatter Plot 2개 완료\n\n');

% ============================================================
%% 5. SVM 모델 학습 및 평가
% ============================================================
fprintf('[5/5] SVM 모델 학습 및 평가 중...\n');

% ── 5-1. SVM (RBF Kernel, One-vs-One Multiclass) ──────────────
fprintf('  >> SVM (RBF Kernel) 학습 중 (잠시 기다려 주세요)...\n');
t_svm = tic;
svm_template = templateSVM('KernelFunction', 'rbf', ...
                            'BoxConstraint',  1, ...
                            'KernelScale',    'auto', ...
                            'Standardize',    false);
mdl_svm = fitcecoc(X_tr, y_train, ...
                   'Learners', svm_template, ...
                   'Coding',   'onevsone');
fprintf('  >> 학습 완료 (%.1fs)\n\n', toc(t_svm));

% 예측
y_pred = predict(mdl_svm, X_te);

% ── 5-2. Confusion Matrix ─────────────────────────────────────
figure('Name', 'SVM Confusion Matrix', ...
       'Position', [150, 150, 600, 500], 'Color', 'white');

cm = confusionmat(y_test, y_pred, 'Order', [1,2,3]);
confusionchart(cm, class_names, ...
    'Title',         'SVM (RBF) – Confusion Matrix', ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized');
set(gca, 'FontSize', 12);

% ── 5-3. Classification Report ────────────────────────────────
fprintf('\n==============================\n');
fprintf('  Classification Report: SVM (RBF)\n');
fprintf('==============================\n');
fprintf('%-10s %9s %9s %9s %9s\n', 'Class','Precision','Recall','F1-Score','Support');
fprintf('%s\n', repmat('-', 1, 52));

all_prec = zeros(1,3);  all_rec = zeros(1,3);  all_f1 = zeros(1,3);
for k = 1:3
    TP   = sum(y_pred == k & y_test == k);
    FP   = sum(y_pred == k & y_test ~= k);
    FN   = sum(y_pred ~= k & y_test == k);
    prec = TP / max(TP+FP, 1);
    rec  = TP / max(TP+FN, 1);
    f1   = 2*prec*rec / max(prec+rec, 1e-9);
    sup  = sum(y_test == k);
    fprintf('%-10s %9.3f %9.3f %9.3f %9d\n', class_names{k}, prec, rec, f1, sup);
    all_prec(k)=prec;  all_rec(k)=rec;  all_f1(k)=f1;
end
fprintf('%s\n', repmat('-', 1, 52));
fprintf('%-10s                           %9.3f\n', 'Accuracy',  mean(y_pred==y_test));
fprintf('%-10s %9.3f %9.3f %9.3f\n\n', 'Macro Avg', ...
        mean(all_prec), mean(all_rec), mean(all_f1));

% ── 5-4. ROC Curve (One-vs-Rest, Multi-class) ─────────────────
[~, score_svm] = predict(mdl_svm, X_te);

figure('Name', 'SVM ROC Curve', ...
       'Position', [150, 150, 700, 580], 'Color', 'white');
hold on;

line_styles = {'-', '--', ':'};
auc_vals    = zeros(1,3);

for k = 1:3
    y_bin = double(y_test == k);
    sc    = score_svm(:, k);
    [fpr, tpr, ~, auc] = perfcurve(y_bin, sc, 1);
    auc_vals(k) = auc;
    plot(fpr, tpr, line_styles{k}, 'Color', PALETTE(k,:), ...
        'LineWidth', 2.5, ...
        'DisplayName', sprintf('%s  (AUC = %.3f)', class_names{k}, auc));
end

plot([0,1],[0,1],'k--','LineWidth',1,'HandleVisibility','off');
hold off;

xlabel('False Positive Rate', 'FontSize', 13);
ylabel('True Positive Rate',  'FontSize', 13);
title(sprintf('SVM (RBF) – ROC Curve\nMacro AUC = %.3f', mean(auc_vals)), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 11);
set(gca, 'FontSize', 12, 'Color', BG_COLOR, 'Box', 'off');
grid on;
xlim([0,1]); ylim([0,1]);

% ── 5-5. 최종 요약 출력 ───────────────────────────────────────
acc = mean(y_pred == y_test);
fprintf('╔══════════════════════════════════════╗\n');
fprintf('║        최종 모델 성능 요약             ║\n');
fprintf('╠══════════════════════════════════════╣\n');
fprintf('║  SVM (RBF)   Accuracy  : %.4f      ║\n', acc);
fprintf('║  SVM (RBF)   Macro AUC : %.4f      ║\n', mean(auc_vals));
fprintf('╚══════════════════════════════════════╝\n\n');
fprintf('모든 분석 완료! 그래프를 확인하세요.\n');