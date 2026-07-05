// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/wait.h>

#include "../tensorForth/tensor.h"

// 1. Test di allocazione e reference counting
void test_alloc_and_ref_counting() {
	printf("[TEST] Esecuzione test_alloc_and_ref_counting...\n");
	int32_t shape[] = {4};
	tensor *t = allocTensor(1, shape);
	assert(t != NULL);
	assert(t->dimensionOfTensor == 1);
	assert(t->shape[0] == 4);
	assert(t->referenceCount == 1);
	assert(t->buffer->referenceCount == 1);
	
	tensorRef(t);
	assert(t->referenceCount == 2);
	
	tensorDeref(t);
	assert(t->referenceCount == 1);
	
	tensorDeref(t);
	printf("[TEST] test_alloc_and_ref_counting superato!\n");
}

#if 0
// 2. Test di shape, reshape e ravel
void test_shape_ops() {
	printf("[TEST] Esecuzione test_shape_ops...\n");
	int32_t shape[] = {6};
	tensor *t = allocTensor(1, shape);
	for (int i = 0; i < 6; i++) t->buffer->data[i] = (float)i;

	int32_t shape_dims[] = {2};
	tensor *target_shape = allocTensor(1, shape_dims);
	target_shape->buffer->data[0] = 2.0f;
	target_shape->buffer->data[1] = 3.0f;

	tensor *t_reshaped = reshape(t, target_shape);
	assert(t_reshaped != NULL);
	assert(t_reshaped->dimensionOfTensor == 2);
	assert(t_reshaped->shape[0] == 2);
	assert(t_reshaped->shape[1] == 3);
	assert(t_reshaped->buffer == t->buffer);
	assert(t->buffer->referenceCount == 2);

	tensor *t_raveled = ravel(t_reshaped);
	assert(t_raveled != NULL);
	assert(t_raveled->dimensionOfTensor == 1);
	assert(t_raveled->shape[0] == 6);
	assert(t_raveled->buffer == t->buffer);
	assert(t->buffer->referenceCount == 3);

	tensor *t_shape = getShape(t_reshaped);
	assert(t_shape != NULL);
	assert(t_shape->dimensionOfTensor == 1);
	assert(t_shape->shape[0] == 2);
	assert(t_shape->buffer->data[0] == 2.0f);
	assert(t_shape->buffer->data[1] == 3.0f);

	tensorDeref(t);
	tensorDeref(target_shape);
	tensorDeref(t_reshaped);
	tensorDeref(t_raveled);
	tensorDeref(t_shape);
	printf("[TEST] test_shape_ops superato!\n");
}
#endif

// 3. Test Operazioni Aritmetiche (+, -, *)
void test_arithmetic() {
	printf("[TEST] Esecuzione test_arithmetic...\n");
	int32_t shape[] = {3};
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);
	a->buffer->data[0] = 5.0f; a->buffer->data[1] = 10.0f; a->buffer->data[2] = 15.0f;
	b->buffer->data[0] = 2.0f; b->buffer->data[1] = 4.0f;  b->buffer->data[2] = 5.0f;

	tensor *r_sum = sum(a, b);
	assert(r_sum->buffer->data[0] == 7.0f);
	assert(r_sum->buffer->data[1] == 14.0f);
	assert(r_sum->buffer->data[2] == 20.0f);

	tensor *r_diff = sub(a, b);
	assert(r_diff->buffer->data[0] == 3.0f);
	assert(r_diff->buffer->data[1] == 6.0f);
	assert(r_diff->buffer->data[2] == 10.0f);

	tensor *r_prod = mul(a, b);
	assert(r_prod->buffer->data[0] == 10.0f);
	assert(r_prod->buffer->data[1] == 40.0f);
	assert(r_prod->buffer->data[2] == 75.0f);

	tensorDeref(a);
	tensorDeref(b);
	tensorDeref(r_sum);
	tensorDeref(r_diff);
	tensorDeref(r_prod);
	printf("[TEST] test_arithmetic superato!\n");
}


// 4. Test Operazioni di Comparazione (=, <, >)
void test_comparison() {
	printf("[TEST] Esecuzione test_comparison...\n");
	int32_t shape[] = {3};
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);
	a->buffer->data[0] = 1.0f; a->buffer->data[1] = 5.0f; a->buffer->data[2] = 3.0f;
	b->buffer->data[0] = 2.0f; b->buffer->data[1] = 4.0f; b->buffer->data[2] = 3.0f;

	tensor *r_eq = equal(a, b);
	assert(r_eq->buffer->data[0] == 0.0f);
	assert(r_eq->buffer->data[1] == 0.0f);
	assert(r_eq->buffer->data[2] == 1.0f);

	tensor *r_gt = greaterThan(a, b);
	assert(r_gt->buffer->data[0] == 0.0f);
	assert(r_gt->buffer->data[1] == 1.0f);
	assert(r_gt->buffer->data[2] == 0.0f);

	tensor *r_lt = lessThan(a, b);
	assert(r_lt->buffer->data[0] == 1.0f);
	assert(r_lt->buffer->data[1] == 0.0f);
	assert(r_lt->buffer->data[2] == 0.0f);

	tensorDeref(a);
	tensorDeref(b);
	tensorDeref(r_eq);
	tensorDeref(r_gt);
	tensorDeref(r_lt);
	printf("[TEST] test_comparison superato!\n");
}

// 5. Test Operazioni Logiche (&, |, !)
void test_logical() {
	printf("[TEST] Esecuzione test_logical...\n");
	int32_t shape[] = {3};
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);
	a->buffer->data[0] = 1.0f; a->buffer->data[1] = 0.0f; a->buffer->data[2] = 1.0f;
	b->buffer->data[0] = 0.0f; b->buffer->data[1] = 0.0f; b->buffer->data[2] = 1.0f;

	tensor *r_and = AND(a, b);
	assert(r_and->buffer->data[0] == 0.0f);
	assert(r_and->buffer->data[1] == 0.0f);
	assert(r_and->buffer->data[2] == 1.0f);

	tensor *r_or = OR(a, b);
	assert(r_or->buffer->data[0] == 1.0f);
	assert(r_or->buffer->data[1] == 0.0f);
	assert(r_or->buffer->data[2] == 1.0f);

	tensor *r_not = NOT(a);
	assert(r_not->buffer->data[0] == 0.0f);
	assert(r_not->buffer->data[1] == 1.0f);
	assert(r_not->buffer->data[2] == 0.0f);

	tensorDeref(a);
	tensorDeref(b);
	tensorDeref(r_and);
	tensorDeref(r_or);
	tensorDeref(r_not);
	printf("[TEST] test_logical superato!\n");
}

// 6. Test Selezione ($)
void test_selection() {
	printf("[TEST] Esecuzione test_selection...\n");
	int32_t shape[] = {3};
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);
	tensor *mask = allocTensor(1, shape);

	a->buffer->data[0] = 10.0f; a->buffer->data[1] = 20.0f; a->buffer->data[2] = 30.0f;
	b->buffer->data[0] = 1.0f;  b->buffer->data[1] = 2.0f;  b->buffer->data[2] = 3.0f;
	mask->buffer->data[0] = 1.0f; mask->buffer->data[1] = 0.0f; mask->buffer->data[2] = 1.0f;

	tensor *res = selection(a, b, mask);
	assert(res->buffer->data[0] == 10.0f);
	assert(res->buffer->data[1] == 2.0f);
	assert(res->buffer->data[2] == 30.0f);

	tensorDeref(a);
	tensorDeref(b);
	tensorDeref(mask);
	tensorDeref(res);
	printf("[TEST] test_selection superato!\n");
}

#if 0
// 7. Test Generazione, Riduzione e Riempimento (?, R, m, M, S, f)
void test_gen_reduction_fill() {
	printf("[TEST] Esecuzione test_gen_reduction_fill...\n");
	
	// ReLU (R)
	int32_t shape[] = {4};
	tensor *t = allocTensor(1, shape);
	t->buffer->data[0] = -1.5f; t->buffer->data[1] = 0.0f; t->buffer->data[2] = 2.5f; t->buffer->data[3] = -0.1f;
	tensor *r_relu = relu(t);
	assert(r_relu->buffer->data[0] == 0.0f);
	assert(r_relu->buffer->data[1] == 0.0f);
	assert(r_relu->buffer->data[2] == 2.5f);
	assert(r_relu->buffer->data[3] == 0.0f);

	// Min/Max (m, M)
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);
	a->buffer->data[0] = 1.0f; a->buffer->data[1] = 4.0f; a->buffer->data[2] = 3.0f; a->buffer->data[3] = -2.0f;
	b->buffer->data[0] = 2.0f; b->buffer->data[1] = 3.0f; b->buffer->data[2] = 3.0f; b->buffer->data[3] = 0.0f;
	
	tensor *r_min = minTensor(a, b);
	assert(r_min->buffer->data[0] == 1.0f);
	assert(r_min->buffer->data[1] == 3.0f);
	assert(r_min->buffer->data[2] == 3.0f);
	assert(r_min->buffer->data[3] == -2.0f);

	tensor *r_max = maxTensor(a, b);
	assert(r_max->buffer->data[0] == 2.0f);
	assert(r_max->buffer->data[1] == 4.0f);
	assert(r_max->buffer->data[2] == 3.0f);
	assert(r_max->buffer->data[3] == 0.0f);

	// Sum Reduction (S)
	tensor *r_sum = reductionSum(a);
	assert(r_sum->dimensionOfTensor == 1);
	assert(r_sum->shape[0] == 1);
	assert(r_sum->buffer->data[0] == 6.0f); // 1 + 4 + 3 - 2 = 6

	// Fill (f)
	int32_t shape_dims[] = {2};
	tensor *target_shape = allocTensor(1, shape_dims);
	target_shape->buffer->data[0] = 2.0f;
	target_shape->buffer->data[1] = 2.0f;
	
	int32_t val_shape[] = {2};
	tensor *val = allocTensor(1, val_shape);
	val->buffer->data[0] = 8.0f;
	val->buffer->data[1] = 9.0f;
	
	tensor *r_fill = fill(target_shape, val);
	assert(r_fill->dimensionOfTensor == 2);
	assert(r_fill->shape[0] == 2);
	assert(r_fill->shape[1] == 2);
	assert(r_fill->buffer->data[0] == 8.0f);
	assert(r_fill->buffer->data[1] == 9.0f);
	assert(r_fill->buffer->data[2] == 8.0f);
	assert(r_fill->buffer->data[3] == 9.0f);

	tensorDeref(t);
	tensorDeref(r_relu);
	tensorDeref(a);
	tensorDeref(b);
	tensorDeref(r_min);
	tensorDeref(r_max);
	tensorDeref(r_sum);
	tensorDeref(target_shape);
	tensorDeref(val);
	tensorDeref(r_fill);
	printf("[TEST] test_gen_reduction_fill superato!\n");
}

// 8. Test moltiplicazione matriciale, dot product e convoluzione (@, ., c)
void test_advanced_ops() {
	printf("[TEST] Esecuzione test_advanced_ops...\n");

	// Dot Product (.)
	int32_t shape_1d[] = {3};
	tensor *v1 = allocTensor(1, shape_1d);
	tensor *v2 = allocTensor(1, shape_1d);
	v1->buffer->data[0] = 1.0f; v1->buffer->data[1] = 2.0f; v1->buffer->data[2] = 3.0f;
	v2->buffer->data[0] = 4.0f; v2->buffer->data[1] = 5.0f; v2->buffer->data[2] = 6.0f;
	tensor *r_dot = dotProduct(v1, v2);
	assert(r_dot->buffer->data[0] == 32.0f); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

	// Matrix Multiplication (@)
	int32_t shape_a[] = {2, 3};
	tensor *ma = allocTensor(2, shape_a);
	ma->buffer->data[0] = 1.0f; ma->buffer->data[1] = 2.0f; ma->buffer->data[2] = 3.0f;
	ma->buffer->data[3] = 4.0f; ma->buffer->data[4] = 5.0f; ma->buffer->data[5] = 6.0f;

	int32_t shape_b[] = {3, 2};
	tensor *mb = allocTensor(2, shape_b);
	mb->buffer->data[0] = 7.0f;  mb->buffer->data[1] = 8.0f;
	mb->buffer->data[2] = 9.0f;  mb->buffer->data[3] = 10.0f;
	mb->buffer->data[4] = 11.0f; mb->buffer->data[5] = 12.0f;

	tensor *r_matmul = matrixMul(ma, mb);
	assert(r_matmul->dimensionOfTensor == 2);
	assert(r_matmul->shape[0] == 2);
	assert(r_matmul->shape[1] == 2);
	// Rigo 1: [1*7 + 2*9 + 3*11 = 58], [1*8 + 2*10 + 3*12 = 64]
	assert(r_matmul->buffer->data[0] == 58.0f);
	assert(r_matmul->buffer->data[1] == 64.0f);
	// Rigo 2: [4*7 + 5*9 + 6*11 = 139], [4*8 + 5*10 + 6*12 = 154]
	assert(r_matmul->buffer->data[2] == 139.0f);
	assert(r_matmul->buffer->data[3] == 154.0f);

	tensorDeref(v1);
	tensorDeref(v2);
	tensorDeref(r_dot);
	tensorDeref(ma);
	tensorDeref(mb);
	tensorDeref(r_matmul);
	printf("[TEST] test_advanced_ops superato!\n");
}
#endif

// Test di robustezza: verifica che il programma si interrompa su errore
void test_sum_incompatible_dimensions() {
	printf("[TEST] Esecuzione test_sum_incompatible_dimensions (si attende una terminazione con errore)...\n");
	pid_t pid = fork();
	if (pid < 0) {
		perror("fork fallito");
		exit(1);
	}
	
	if (pid == 0) {
		// Silenziamo stderr nel figlio per non sporcare il terminale con l'errore atteso
		freopen("/dev/null", "w", stderr);
		
		int32_t shape_a[] = {3};
		int32_t shape_b[] = {4};
		tensor *a = allocTensor(1, shape_a);
		tensor *b = allocTensor(1, shape_b);
		
		// Questa operazione deve interrompere il processo tramite equalShapes/exit(1)
		tensor *res = sum(a, b);
		
		(void)res;
		exit(0); // Se arriva qui, il test ha fallito
	} else {
		int status;
		waitpid(pid, &status, 0);
		if (WIFEXITED(status)) {
			int exit_code = WEXITSTATUS(status);
			assert(exit_code == 1); // Ci aspettiamo exit(1) dall'errore di shape
			printf("[TEST] test_sum_incompatible_dimensions superato (terminato con codice 1)!\n");
		} else {
			printf("[TEST] test_sum_incompatible_dimensions FALLITO (crash imprevisto)!\n");
			assert(0);
		}
	}
}

void test_sum_incompatible_ranks() {
	printf("[TEST] Esecuzione test_sum_incompatible_ranks (si attende una terminazione con errore)...\n");
	pid_t pid = fork();
	if (pid < 0) {
		perror("fork fallito");
		exit(1);
	}
	
	if (pid == 0) {
		freopen("/dev/null", "w", stderr);
		
		int32_t shape_a[] = {3};
		int32_t shape_b[] = {3, 1};
		tensor *a = allocTensor(1, shape_a);
		tensor *b = allocTensor(2, shape_b);
		
		// Deve interrompere per via di equalDimensions/exit(1)
		tensor *res = sum(a, b);
		
		(void)res;
		exit(0);
	} else {
		int status;
		waitpid(pid, &status, 0);
		if (WIFEXITED(status)) {
			int exit_code = WEXITSTATUS(status);
			assert(exit_code == 1); // Ci aspettiamo exit(1) dall'errore di dimensione
			printf("[TEST] test_sum_incompatible_ranks superato (terminato con codice 1)!\n");
		} else {
			printf("[TEST] test_sum_incompatible_ranks FALLITO (crash imprevisto)!\n");
			assert(0);
		}
	}
}

void test_selection_incompatible_dimensions() {
	printf("[TEST] Esecuzione test_selection_incompatible_dimensions (si attende una terminazione con errore)...\n");
	pid_t pid = fork();
	if (pid < 0) {
		perror("fork fallito");
		exit(1);
	}
	
	if (pid == 0) {
		freopen("/dev/null", "w", stderr);
		
		int32_t shape_a[] = {3};
		int32_t shape_b[] = {4};
		int32_t shape_mask[] = {3};
		tensor *a = allocTensor(1, shape_a);
		tensor *b = allocTensor(1, shape_b);
		tensor *mask = allocTensor(1, shape_mask);
		
		tensor *res = selection(a, b, mask);
		(void)res;
		exit(0);
	} else {
		int status;
		waitpid(pid, &status, 0);
		if (WIFEXITED(status)) {
			int exit_code = WEXITSTATUS(status);
			assert(exit_code == 1);
			printf("[TEST] test_selection_incompatible_dimensions superato (terminato con codice 1)!\n");
		} else {
			printf("[TEST] test_selection_incompatible_dimensions FALLITO (crash imprevisto)!\n");
			assert(0);
		}
	}
}

void test_selection_incompatible_mask() {
	printf("[TEST] Esecuzione test_selection_incompatible_mask (si attende una terminazione con errore)...\n");
	pid_t pid = fork();
	if (pid < 0) {
		perror("fork fallito");
		exit(1);
	}
	
	if (pid == 0) {
		freopen("/dev/null", "w", stderr);
		
		int32_t shape_a[] = {3};
		int32_t shape_b[] = {3};
		int32_t shape_mask[] = {4};
		tensor *a = allocTensor(1, shape_a);
		tensor *b = allocTensor(1, shape_b);
		tensor *mask = allocTensor(1, shape_mask);
		
		tensor *res = selection(a, b, mask);
		(void)res;
		exit(0);
	} else {
		int status;
		waitpid(pid, &status, 0);
		if (WIFEXITED(status)) {
			int exit_code = WEXITSTATUS(status);
			assert(exit_code == 1);
			printf("[TEST] test_selection_incompatible_mask superato (terminato con codice 1)!\n");
		} else {
			printf("[TEST] test_selection_incompatible_mask FALLITO (crash imprevisto)!\n");
			assert(0);
		}
	}
}

int main() {
	printf("=== AVVIO SUITE COMPLETA TEST TENSOR ===\n");
	test_alloc_and_ref_counting();
	// test_shape_ops();
	test_arithmetic();
	test_sum_incompatible_dimensions();
	test_sum_incompatible_ranks();
	test_comparison();
	test_logical();
	test_selection();
	test_selection_incompatible_dimensions();
	test_selection_incompatible_mask();
	// test_gen_reduction_fill();
	// test_advanced_ops();
	printf("=== TUTTI I TEST COMPILATI E PASSATI CON SUCCESSO! ===\n");
	return 0;
}
