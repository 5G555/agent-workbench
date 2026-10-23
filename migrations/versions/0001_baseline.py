"""baseline：既有表结构（由 pg_dump 生成，空库可重建）

来源：本地库 pg_dump --schema-only 的结构快照。
现有库已经执行过 alembic stamp head，因此 upgrade 不会再跑一遍。

Revision ID: 0001_baseline
Revises: None
"""

from __future__ import annotations

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE public.crawl_pages (\n    id bigint NOT NULL,\n    task_id bigint NOT NULL,\n    url character varying(1024) NOT NULL,\n    final_url character varying(1024) DEFAULT ''::character varying NOT NULL,\n    title character varying(512) DEFAULT ''::character varying NOT NULL,\n    status integer DEFAULT 0 NOT NULL,\n    mode character varying(16) DEFAULT 'http'::character varying NOT NULL,\n    char_count integer DEFAULT 0 NOT NULL,\n    content text DEFAULT ''::text NOT NULL,\n    error text DEFAULT ''::text NOT NULL,\n    fetched_at timestamp with time zone DEFAULT now() NOT NULL,\n    html text DEFAULT ''::text NOT NULL\n)"
    )
    op.execute(
        "CREATE TABLE public.crawl_tasks (\n    id bigint NOT NULL,\n    tenant_id character varying(64) DEFAULT 'tenant_default'::character varying NOT NULL,\n    user_id character varying(64) DEFAULT ''::character varying NOT NULL,\n    goal character varying(1024) DEFAULT ''::character varying NOT NULL,\n    entry_url character varying(1024) DEFAULT ''::character varying NOT NULL,\n    status character varying(16) DEFAULT 'running'::character varying NOT NULL,\n    pages_fetched integer DEFAULT 0 NOT NULL,\n    records_extracted integer DEFAULT 0 NOT NULL,\n    strategy character varying(32) DEFAULT ''::character varying NOT NULL,\n    stop_reason character varying(64) DEFAULT ''::character varying NOT NULL,\n    error text DEFAULT ''::text NOT NULL,\n    started_at timestamp with time zone DEFAULT now() NOT NULL,\n    finished_at timestamp with time zone\n)"
    )
    op.execute(
        "CREATE VIEW public.ana_crawl_page AS\n SELECT p.id AS page_id,\n    p.task_id,\n    t.tenant_id,\n    p.url,\n    p.title,\n    p.status,\n    p.mode,\n    p.char_count,\n    p.fetched_at\n   FROM (public.crawl_pages p\n     JOIN public.crawl_tasks t ON ((t.id = p.task_id)))\n  WHERE ((t.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE TABLE public.crawl_records (\n    id bigint NOT NULL,\n    task_id bigint NOT NULL,\n    page_url character varying(1024) DEFAULT ''::character varying NOT NULL,\n    strategy character varying(32) DEFAULT ''::character varying NOT NULL,\n    data text DEFAULT '{}'::text NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL\n)"
    )
    op.execute(
        "CREATE VIEW public.ana_crawl_record AS\n SELECT r.id AS record_id,\n    r.task_id,\n    t.tenant_id,\n    r.page_url,\n    r.strategy,\n    r.data,\n    r.created_at\n   FROM (public.crawl_records r\n     JOIN public.crawl_tasks t ON ((t.id = r.task_id)))\n  WHERE ((t.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE VIEW public.ana_crawl_task AS\n SELECT t.id AS task_id,\n    t.tenant_id,\n    t.user_id,\n    t.goal,\n    t.entry_url,\n    t.status,\n    t.pages_fetched,\n    t.records_extracted,\n    t.strategy,\n    t.stop_reason,\n    t.started_at,\n    t.finished_at\n   FROM public.crawl_tasks t\n  WHERE ((t.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE TABLE public.faq_entries (\n    id bigint NOT NULL,\n    subject character varying(64) DEFAULT ''::character varying NOT NULL,\n    question character varying(1000) NOT NULL,\n    answer character varying(4000) NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL,\n    tenant_id character varying(64) DEFAULT 'tenant_default'::character varying NOT NULL,\n    department character varying(64) DEFAULT '*'::character varying NOT NULL\n)"
    )
    op.execute(
        "CREATE VIEW public.ana_faq_entry AS\n SELECT f.id AS faq_id,\n    f.tenant_id,\n    f.department,\n    f.subject,\n    f.question,\n    f.created_at\n   FROM public.faq_entries f\n  WHERE ((f.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE TABLE public.qa_feedback (\n    id bigint NOT NULL,\n    session_id character varying(128) NOT NULL,\n    user_id character varying(128) DEFAULT ''::character varying NOT NULL,\n    rating smallint NOT NULL,\n    answer_hash character varying(64) NOT NULL,\n    comment text DEFAULT ''::text NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL\n)"
    )
    op.execute(
        "CREATE TABLE public.users (\n    id bigint NOT NULL,\n    username character varying(64) NOT NULL,\n    password_hash text NOT NULL,\n    role character varying(16) DEFAULT 'user'::character varying NOT NULL,\n    tenant_id character varying(64) DEFAULT 'tenant_default'::character varying NOT NULL,\n    is_active boolean DEFAULT true NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL,\n    department character varying(64) DEFAULT '*'::character varying NOT NULL\n)"
    )
    op.execute(
        "CREATE VIEW public.ana_qa_feedback AS\n SELECT f.id AS feedback_id,\n    f.session_id,\n    u.tenant_id,\n    u.username,\n    u.department,\n    f.rating,\n    f.comment,\n    f.created_at\n   FROM (public.qa_feedback f\n     JOIN public.users u ON (((u.id)::text = (f.user_id)::text)))\n  WHERE ((u.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE TABLE public.qa_sessions (\n    session_id character varying(128) NOT NULL,\n    user_id character varying(128) DEFAULT ''::character varying NOT NULL,\n    title character varying(128) DEFAULT ''::character varying NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL,\n    updated_at timestamp with time zone DEFAULT now() NOT NULL\n)"
    )
    op.execute(
        "CREATE VIEW public.ana_qa_session AS\n SELECT s.session_id,\n    u.tenant_id,\n    u.username,\n    u.department,\n    s.title,\n    s.created_at,\n    s.updated_at\n   FROM (public.qa_sessions s\n     JOIN public.users u ON (((u.id)::text = (s.user_id)::text)))\n  WHERE ((u.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE TABLE public.qa_messages (\n    id bigint NOT NULL,\n    session_id character varying(128) NOT NULL,\n    user_id character varying(128) DEFAULT ''::character varying NOT NULL,\n    role character varying(16) NOT NULL,\n    content text NOT NULL,\n    answer_mode character varying(16) DEFAULT ''::character varying NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL\n)"
    )
    op.execute(
        "CREATE VIEW public.ana_qa_turn AS\n SELECT m.id AS message_id,\n    m.session_id,\n    u.tenant_id,\n    u.username,\n    u.department,\n    m.role,\n    m.answer_mode,\n    m.created_at\n   FROM (public.qa_messages m\n     JOIN public.users u ON (((u.id)::text = (m.user_id)::text)))\n  WHERE ((u.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE VIEW public.ana_user AS\n SELECT u.id AS user_id,\n    u.tenant_id,\n    u.username,\n    u.role,\n    u.department,\n    u.is_active,\n    u.created_at\n   FROM public.users u\n  WHERE ((u.tenant_id)::text = current_setting('app.tenant_id'::text, true))"
    )
    op.execute(
        "CREATE SEQUENCE public.crawl_pages_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.crawl_pages_id_seq OWNED BY public.crawl_pages.id")
    op.execute(
        "CREATE SEQUENCE public.crawl_records_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.crawl_records_id_seq OWNED BY public.crawl_records.id")
    op.execute(
        "CREATE SEQUENCE public.crawl_tasks_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.crawl_tasks_id_seq OWNED BY public.crawl_tasks.id")
    op.execute(
        "CREATE SEQUENCE public.faq_entries_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.faq_entries_id_seq OWNED BY public.faq_entries.id")
    op.execute(
        "CREATE TABLE public.ops_audit (\n    id bigint NOT NULL,\n    tenant_id character varying(64) DEFAULT 'tenant_default'::character varying NOT NULL,\n    requester character varying(64) DEFAULT ''::character varying NOT NULL,\n    tool character varying(64) DEFAULT ''::character varying NOT NULL,\n    args text DEFAULT '{}'::text NOT NULL,\n    risk character varying(16) DEFAULT ''::character varying NOT NULL,\n    decision character varying(24) DEFAULT ''::character varying NOT NULL,\n    reasons text DEFAULT ''::text NOT NULL,\n    approval_id character varying(64) DEFAULT ''::character varying NOT NULL,\n    approver character varying(64) DEFAULT ''::character varying NOT NULL,\n    self_approved boolean DEFAULT false NOT NULL,\n    snapshot character varying(1024) DEFAULT ''::character varying NOT NULL,\n    status character varying(24) DEFAULT 'executing'::character varying NOT NULL,\n    result text DEFAULT ''::text NOT NULL,\n    created_at timestamp with time zone DEFAULT now() NOT NULL,\n    finished_at timestamp with time zone\n)"
    )
    op.execute(
        "CREATE SEQUENCE public.ops_audit_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.ops_audit_id_seq OWNED BY public.ops_audit.id")
    op.execute(
        "CREATE SEQUENCE public.qa_feedback_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.qa_feedback_id_seq OWNED BY public.qa_feedback.id")
    op.execute(
        "CREATE SEQUENCE public.qa_messages_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.qa_messages_id_seq OWNED BY public.qa_messages.id")
    op.execute(
        "CREATE TABLE public.route_feedback (\n    id bigint NOT NULL,\n    tenant_id character varying(64) NOT NULL,\n    user_id character varying(64) NOT NULL,\n    session_id character varying(128) DEFAULT ''::character varying,\n    question text NOT NULL,\n    routed_agent character varying(32) NOT NULL,\n    corrected_agent character varying(32) NOT NULL,\n    confidence double precision DEFAULT 0,\n    reason text DEFAULT ''::text,\n    created_at timestamp with time zone DEFAULT now() NOT NULL\n)"
    )
    op.execute(
        "CREATE SEQUENCE public.route_feedback_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.route_feedback_id_seq OWNED BY public.route_feedback.id")
    op.execute(
        "CREATE SEQUENCE public.users_id_seq\n    START WITH 1\n    INCREMENT BY 1\n    NO MINVALUE\n    NO MAXVALUE\n    CACHE 1"
    )
    op.execute("ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id")
    op.execute(
        "ALTER TABLE ONLY public.crawl_pages\n    ADD CONSTRAINT crawl_pages_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.crawl_pages\n    ADD CONSTRAINT crawl_pages_task_id_url_key UNIQUE (task_id, url)"
    )
    op.execute(
        "ALTER TABLE ONLY public.crawl_records\n    ADD CONSTRAINT crawl_records_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.crawl_tasks\n    ADD CONSTRAINT crawl_tasks_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.faq_entries\n    ADD CONSTRAINT faq_entries_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.ops_audit\n    ADD CONSTRAINT ops_audit_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.qa_feedback\n    ADD CONSTRAINT qa_feedback_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.qa_feedback\n    ADD CONSTRAINT qa_feedback_session_id_answer_hash_key UNIQUE (session_id, answer_hash)"
    )
    op.execute(
        "ALTER TABLE ONLY public.qa_messages\n    ADD CONSTRAINT qa_messages_pkey PRIMARY KEY (id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.qa_sessions\n    ADD CONSTRAINT qa_sessions_pkey PRIMARY KEY (session_id)"
    )
    op.execute(
        "ALTER TABLE ONLY public.route_feedback\n    ADD CONSTRAINT route_feedback_pkey PRIMARY KEY (id)"
    )
    op.execute("ALTER TABLE ONLY public.users\n    ADD CONSTRAINT users_pkey PRIMARY KEY (id)")
    op.execute(
        "ALTER TABLE ONLY public.users\n    ADD CONSTRAINT users_username_key UNIQUE (username)"
    )
    op.execute(
        "CREATE INDEX idx_route_feedback_tenant ON public.route_feedback USING btree (tenant_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_crawl_records_task ON public.crawl_records USING btree (task_id, id)"
    )
    op.execute(
        "CREATE INDEX ix_crawl_tasks_tenant ON public.crawl_tasks USING btree (tenant_id, id DESC)"
    )
    op.execute(
        "CREATE INDEX ix_ops_audit_tenant_time ON public.ops_audit USING btree (tenant_id, id DESC)"
    )
    op.execute("CREATE INDEX ix_ops_audit_tool ON public.ops_audit USING btree (tool, id DESC)")
    op.execute("CREATE INDEX ix_qa_feedback_session ON public.qa_feedback USING btree (session_id)")
    op.execute(
        "CREATE INDEX ix_qa_messages_session ON public.qa_messages USING btree (session_id, id)"
    )
    op.execute(
        "CREATE UNIQUE INDEX ux_faq_entries_scope_question ON public.faq_entries USING btree (tenant_id, department, question)"
    )
    op.execute(
        "ALTER TABLE ONLY public.crawl_pages\n    ADD CONSTRAINT crawl_pages_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.crawl_tasks(id) ON DELETE CASCADE"
    )
    op.execute(
        "ALTER TABLE ONLY public.crawl_records\n    ADD CONSTRAINT crawl_records_task_id_fkey FOREIGN KEY (task_id) REFERENCES public.crawl_tasks(id) ON DELETE CASCADE"
    )


def downgrade() -> None:
    """baseline 不撤销（重建请用空库 upgrade head）。"""
